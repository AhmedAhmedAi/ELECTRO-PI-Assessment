"""Pipeline orchestration: question -> semantic cache -> retrieve -> answer,
plus the assessment examples, threshold tuning, and result artifacts.
Entry point: analysis.ipynb (runs every stage as cells)."""
import time
import warnings

warnings.filterwarnings("ignore")

import config


class Pipeline:
    """Question -> semantic cache -> retrieve -> answer. Loaded once."""

    def __init__(self):
        from services.answerer import REFUSAL, Answerer
        from services.cache import SemanticCache
        from services.retriever import Retriever
        self.retriever = Retriever()
        self.answerer = Answerer()
        self.cache = SemanticCache()
        self.refusal = REFUSAL

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()

        from services.cache import greeting_answer
        greeting = greeting_answer(question)
        if greeting:
            return {
                "question": question, "answer": greeting, "citations": [],
                "refused": False, "contexts": [], "cache_hit": True,
                "diagnostics": {"matched_question": "(greeting pattern)",
                                "similarity": 1.0},
                "seconds": round(time.perf_counter() - t0, 2),
            }

        vector = self.retriever.embedder.embed_query(question)

        hit = self.cache.lookup(vector)
        if hit:
            return {
                "question": question, "answer": hit["answer"],
                "citations": hit["citations"], "refused": hit["answer"] == self.refusal,
                "contexts": [], "cache_hit": True,
                "diagnostics": {"matched_question": hit["question"],
                                "similarity": hit["similarity"]},
                "seconds": round(time.perf_counter() - t0, 2),
            }

        retrieval = self.retriever.retrieve(question, vector)
        if retrieval["status"] == "no_context":
            result = {"answer": self.refusal, "citations": [], "refused": True,
                      "cache_mode": self.answerer.mode, "cached_tokens": 0}
        else:
            result = self.answerer.answer(question, retrieval["contexts"])

        self.cache.store(question, vector, result["answer"], result["citations"])
        return {
            "question": question, **result,
            "contexts": retrieval["contexts"], "cache_hit": False,
            "diagnostics": retrieval["diagnostics"],
            "seconds": round(time.perf_counter() - t0, 2),
        }


def print_result(r: dict):
    print(f"\nQ: {r['question']}")
    print(f"A: {r['answer']}\n")
    for c in r["citations"]:
        print(f"   [{c['n']}] {c['title']} | {c['heading']} — {c['url']}")
    tags = [f"{r['seconds']}s"]
    if r["cache_hit"]:
        d = r["diagnostics"]
        tags.append(f"semantic-cache hit (={d['similarity']} vs {d['matched_question']!r})")
    else:
        tags.append(r.get("cache_mode", ""))
        if r.get("cached_tokens"):
            tags.append(f"{r['cached_tokens']} prompt tokens read from server cache")
        if r["refused"] and "gate" in r["diagnostics"]:
            tags.append(f"guardrail: {r['diagnostics']['gate']}")
    print(f"   ({' · '.join(t for t in tags if t)})")


# The assessment examples: 4 answerable + 1 unanswerable. The pricing one
# was *designed* as the unanswerable trap — then retrieval found real price
# ranges in the chatbot article's FAQ, so it stays as an answered example
# (the system knew the corpus better than its author; see NOTES.md). The CEO
# question is verified absent from the whole corpus: on-topic-sounding, so
# it exercises the guardrail harder than an obviously off-topic question.
EXAMPLES = [
    "When was ElectroPi founded, and how many enterprise clients does it serve?",
    "Why is Arabic voice AI particularly hard to build?",
    "What results did ElectroPi's e-commerce case study achieve?",
    "How much does ElectroPi charge for a chatbot project?",
    "Who is the CEO of ElectroPi?",
]


def fresh_examples(pipe: Pipeline) -> list[dict]:
    """Run the examples through the FULL pipeline: the semantic cache is
    cleared first — an evaluation run must not be answered from cache
    (cache hits carry no retrieval contexts for the judge to check)."""
    config.SEMANTIC_CACHE_FILE.unlink(missing_ok=True)
    from services.cache import SemanticCache
    pipe.cache = SemanticCache()
    return [pipe.ask(q) for q in EXAMPLES]


def save_example_artifacts(results: list[dict], judgments: list[dict]):
    """qa_examples.md / .json + judge_results.json -> results/."""
    from utils.files import save_json

    lines = ["# Section 2 — example Q&As with judge scores", ""]
    for r, j in zip(results, judgments):
        lines += [f"## Q: {r['question']}", "", r["answer"], ""]
        if r["citations"]:
            lines += ["Citations:"] + [
                f"- [{c['n']}] {c['title']} | {c['heading']} — {c['url']}" for c in r["citations"]
            ] + [""]
        if r["refused"]:
            gate = r["diagnostics"].get("gate", "answerer refusal")
            lines += [f"*Refused by guardrail: {gate}*", "",
                      f"Judge: refusal_correct = {j.get('refusal_correct')} — {j['comment']}", ""]
        else:
            lines += [f"Judge: faithfulness {j['faithfulness']}/5 · "
                      f"relevance {j['answer_relevance']}/5 · "
                      f"context precision {j['context_precision']}/5 — {j['comment']}", ""]

    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / "qa_examples.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(results, config.RESULTS_DIR / "qa_examples.json")
    save_json(judgments, config.RESULTS_DIR / "judge_results.json")
    print(f"Saved qa_examples.md/.json + judge_results.json -> {config.RESULTS_DIR}")


def run_examples(pipe: Pipeline):
    from services.judge import judge_examples

    results = fresh_examples(pipe)
    for r in results:
        print_result(r)
    print("\nJudging answers (RAGAS-style rubric, blind to pipeline internals)...")
    judgments = judge_examples(results)
    save_example_artifacts(results, judgments)


# On-topic vs off-topic probes for measuring the dense gate. Off-topic ones
# are chosen adversarially close: generic AI questions the site does NOT cover.
TUNE_ON = [
    "What services does ElectroPi offer?",
    "Tell me about ElectroPi's OCR work",
    "Does ElectroPi build custom machine learning models?",
    "What is ElectroPi's mission?",
    "How do AI chatbots help enterprises?",
    "What did the manufacturing case study deliver?",
]
TUNE_OFF = [
    "What is the capital of France?",
    "How do I cook pasta carbonara?",
    "Who won the 2022 World Cup?",
    "How does a transformer neural network work mathematically?",
    "What is the best GPU for training LLMs?",
    "What's the weather in Cairo today?",
]


def run_tune(pipe: Pipeline):
    from utils.files import save_json

    rows = []
    for label, questions in [("on-topic", TUNE_ON), ("off-topic", TUNE_OFF)]:
        for q in questions:
            vec = pipe.retriever.embedder.embed_query(q)
            best = pipe.retriever.dense_search(vec)[0][1]
            rows.append({"question": q, "label": label, "best_cosine": round(best, 3)})
            print(f"  {best:.3f}  {label:9s}  {q}")

    on = [r["best_cosine"] for r in rows if r["label"] == "on-topic"]
    off = [r["best_cosine"] for r in rows if r["label"] == "off-topic"]
    print(f"\non-topic  min={min(on):.3f}  |  off-topic max={max(off):.3f}"
          f"  |  configured gate={config.DENSE_GATE}")
    save_json({"rows": rows, "on_topic_min": min(on), "off_topic_max": max(off),
               "dense_gate": config.DENSE_GATE, "rerank_gate": config.RERANK_GATE},
              config.RESULTS_DIR / "threshold_tuning.json")
    print(f"Saved -> {config.RESULTS_DIR / 'threshold_tuning.json'}")
