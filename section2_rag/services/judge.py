"""RAGAS-style LLM judge for the example Q&As (same rubric-judge approach
as my Section 3 blind evaluation, adapted to RAG):

  faithfulness       — is every claim in the answer supported by the sources?
  answer_relevance   — does the answer actually address the question?
  context_precision  — how much of the retrieved context was actually needed?

Refusal cases are judged on whether refusing was the right call.
"""
import json

from google import genai
from google.genai import types

import config

RUBRIC_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "faithfulness": {"type": "INTEGER"},
        "answer_relevance": {"type": "INTEGER"},
        "context_precision": {"type": "INTEGER"},
        "unsupported_claims": {"type": "ARRAY", "items": {"type": "STRING"}},
        "comment": {"type": "STRING"},
    },
    "required": ["faithfulness", "answer_relevance", "context_precision", "comment"],
}

REFUSAL_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "refusal_correct": {"type": "BOOLEAN"},
        "comment": {"type": "STRING"},
    },
    "required": ["refusal_correct", "comment"],
}

RUBRIC_PROMPT = """You are evaluating one answer from a RAG system. Score 1-5:

- faithfulness: 5 = every claim is directly supported by the sources below,
  1 = the answer invents facts not in the sources. List any unsupported claims.
- answer_relevance: 5 = fully answers the question asked, 1 = off-topic.
- context_precision: 5 = the retrieved sources were all needed and on-point,
  1 = mostly irrelevant sources were retrieved.

SOURCES GIVEN TO THE SYSTEM:
{sources}

QUESTION: {question}

SYSTEM'S ANSWER: {answer}"""

REFUSAL_PROMPT = """A RAG system limited to a knowledge base about the company
ElectroPi (its website and blog) was asked:

QUESTION: {question}

It refused: "{answer}"

These are the CLOSEST passages retrieval found in the entire knowledge base
(none passed the relevance gate):

{candidates}

Judge only from this evidence, not from assumptions about what a company
website "should" contain: given that these are the nearest passages that
exist, was refusing correct — i.e. do the passages fail to contain the
answer, so answering would have required making something up?"""


def judge_examples(examples: list[dict]) -> list[dict]:
    """Score every example; returns judgments in the same order."""
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    judgments = []
    for ex in examples:
        if ex["refused"]:
            near = ex["diagnostics"].get("rerank", [])
            candidates = "\n\n".join(
                f"- {c['heading']} (relevance {c['relevance']}/10): {c.get('preview', '')}"
                for c in near
            ) or "(nothing passed even the dense similarity gate)"
            prompt = REFUSAL_PROMPT.format(
                question=ex["question"], answer=ex["answer"], candidates=candidates,
            )
            schema = REFUSAL_SCHEMA
        else:
            sources = "\n\n".join(
                f"[{c['n']}] {c['title']} | {c['heading']}\n{c['text']}"
                for c in ex["contexts"]
            )
            prompt = RUBRIC_PROMPT.format(
                sources=sources, question=ex["question"], answer=ex["answer"]
            )
            schema = RUBRIC_SCHEMA

        response = client.models.generate_content(
            model=config.JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0,
            ),
        )
        judgments.append(json.loads(response.text))
    return judgments
