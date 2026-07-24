"""Grounded answering: cached system prompt -> Gemini -> answer + [n] citations.

The system prompt (rules + site map + few-shot examples) is stored once on
Gemini's servers via explicit context caching and referenced by name on
every call — it survives across runs (cache name + expiry persisted in
index/prompt_cache.json). If the prompt is ever under the 1024-token
explicit-cache minimum, we fall back to sending it inline, where Gemini's
implicit caching still applies, and we say which mode ran.
"""
import hashlib
import re
from datetime import datetime, timezone

from google import genai
from google.genai import types
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from utils.files import load_json, save_json

REFUSAL = "I don't have relevant information about that in the ElectroPi knowledge base."

SYSTEM_PROMPT_TEMPLATE = """You are the ElectroPi knowledge assistant. You answer questions
about ElectroPi (electropi.ai) — its services, projects, team, and its blog
articles about AI engineering — using ONLY the numbered sources provided with
each question.

RULES
1. Ground every claim in the sources. After each claim, cite the source it
   came from as [n]. Multiple sources: [1][3].
2. Never use outside knowledge, never guess, never extrapolate numbers or
   names that are not in the sources.
3. If the sources do not contain the answer, reply with exactly:
   "{refusal}"
   Do not apologize at length, do not invent partial answers.
4. Answer directly and concisely: lead with the answer, keep it under 150
   words unless the question genuinely needs more.
5. Quote exact figures (dates, percentages, counts) as written in the sources.

KNOWLEDGE BASE CONTENTS
{site_map}

Use this map when deciding whether a question is answerable: if its topic
does not appear anywhere above, the sources will not contain it — refuse.

EXAMPLE 1
Sources:
[1] About ElectroPi | Our history — "ElectroPi was founded in 2016 by a team
of AI researchers..."
Question: When was ElectroPi founded?
Answer: ElectroPi was founded in 2016 by a team of AI researchers and
software engineers [1].

EXAMPLE 2
Sources:
[1] Services | AI Automation — "We automate business processes using
intelligent agents..."
Question: What is ElectroPi's stock price?
Answer: {refusal}
"""


def build_system_prompt() -> str:
    """Rules + a site map generated from the actual scraped snapshot.

    The map lists every page with its section headings — real grounding
    information (the model can see what the KB does and does not cover),
    and the bulk that puts the prompt over the 1024-token explicit-cache
    minimum without artificial padding.
    """
    pages = load_json(config.SCRAPE_FILE)
    lines = []
    for p in pages:
        headings = [s["heading"] for s in p["sections"] if s["level"] <= 2][:10]
        lines.append(f"- {p['title']} ({p['url']}): " + "; ".join(headings))
    return SYSTEM_PROMPT_TEMPLATE.format(refusal=REFUSAL, site_map="\n".join(lines))

USER_TEMPLATE = """Sources:
{sources}

Question: {question}"""


class Answerer:
    def __init__(self):
        self.client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self.system_prompt = build_system_prompt()
        self.cache_name = self._explicit_cache()
        self.mode = "explicit-cache" if self.cache_name else "implicit-cache"
        kwargs = {"cached_content": self.cache_name} if self.cache_name else {}
        self.llm = ChatGoogleGenerativeAI(
            model=config.ANSWER_MODEL, google_api_key=config.GOOGLE_API_KEY, **kwargs
        )

    def _explicit_cache(self) -> str | None:
        """Create or reuse a server-side cache of the system prompt."""
        prompt_hash = hashlib.md5(self.system_prompt.encode()).hexdigest()
        now = datetime.now(timezone.utc).timestamp()

        if config.PROMPT_CACHE_FILE.exists():
            saved = load_json(config.PROMPT_CACHE_FILE)
            if saved.get("hash") == prompt_hash and saved.get("expires_at", 0) > now + 60:
                return saved["name"]

        try:
            cache = self.client.caches.create(
                model=config.ANSWER_MODEL,
                config=types.CreateCachedContentConfig(
                    system_instruction=self.system_prompt,
                    ttl=f"{config.PROMPT_CACHE_TTL_SECONDS}s",
                ),
            )
        except Exception as e:                       # too small / API hiccup
            print(f"  (explicit cache unavailable — {str(e)[:80]} — using implicit caching)")
            return None

        save_json(
            {"name": cache.name, "hash": prompt_hash,
             "expires_at": now + config.PROMPT_CACHE_TTL_SECONDS,
             "cached_tokens": cache.usage_metadata.total_token_count},
            config.PROMPT_CACHE_FILE,
        )
        return cache.name

    def answer(self, question: str, contexts: list[dict]) -> dict:
        """One grounded answer from the retrieved parent sections."""
        sources = "\n\n".join(
            f"[{c['n']}] {c['title']} | {c['heading']} — {c['url']}\n{c['text']}"
            for c in contexts
        )
        messages = [HumanMessage(USER_TEMPLATE.format(sources=sources, question=question))]
        if not self.cache_name:                      # prompt travels inline instead
            messages.insert(0, SystemMessage(self.system_prompt))

        response = self.llm.invoke(messages)
        text = response.content if isinstance(response.content, str) else "".join(
            b.get("text", "") for b in response.content if isinstance(b, dict)
        )
        text = text.strip()

        cited_ns = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        citations = [
            {"n": c["n"], "title": c["title"], "heading": c["heading"], "url": c["url"]}
            for c in contexts if c["n"] in cited_ns
        ]
        usage = getattr(response, "usage_metadata", None) or {}
        return {
            "answer": text,
            "citations": citations,
            "refused": REFUSAL in text,
            "cache_mode": self.mode,
            "cached_tokens": (usage.get("input_token_details") or {}).get("cache_read", 0),
        }
