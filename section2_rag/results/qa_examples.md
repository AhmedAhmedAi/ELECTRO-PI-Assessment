# Section 2 — example Q&As with judge scores

## Q: When was ElectroPi founded, and how many enterprise clients does it serve?

ElectroPi was founded in 2016 and serves over 130 enterprise clients [1].

Citations:
- [1] Our Story & Mission | About ElectroPi | Built by engineers, for engineers. — https://electropi.ai/about

Judge: faithfulness 5/5 · relevance 5/5 · context precision 5/5 — The answer is fully faithful to the sources, directly answering the question with precise information retrieved from the provided context. All sources retrieved are highly relevant to the query.

## Q: Why is Arabic voice AI particularly hard to build?

Building Arabic voice AI is uniquely difficult for several reasons:

* **Dialect mixing and regional variations:** Systems must handle Egyptian Arabic, Saudi dialects (such as Najdi and Hejazi), Gulf and Levantine dialects, and Modern Standard Arabic (MSA)—often mixing dialects or MSA in the same call [1][2].
* **Code-switching:** Callers frequently switch between Arabic and English mid-sentence [1][2].
* **Noisy environments:** Callers speak from mobile networks, cars, cafes, and streets, which causes off-the-shelf ASR (Automatic Speech Recognition) to degrade quickly [1].
* **Named entities:** Base models frequently mistranscribe local company names, personal names, place names, and Arabic names generally [1][2].
* **Latency budget and morphology:** Arabic morphology increases token counts, making every millisecond critical [1].
* **Off-the-shelf limitations:** Models trained on Western datasets stumble on real MENA traffic and generic global vendors frequently fail on these localization requirements [1][2].

Citations:
- [1] Enterprise Voice AI Solutions in Arabic & English | ElectroPi | Why Arabic voice AI is uniquely hard — https://electropi.ai/blog/voice-ai-solutions
- [2] Voice AI Agent for Customer Service | ElectroPi | Why Arabic Voice AI Matters for Customer Service — https://electropi.ai/blog/voice-ai-agent-for-customer-service

Judge: faithfulness 5/5 · relevance 5/5 · context precision 4/5 — The system's answer is highly faithful to the provided sources, directly addressing the question with accurate details. Every point in the answer is supported by Sources 1 and 2. Context precision is rated 4 because while Sources 1 and 2 were highly relevant and sufficient to answer the question, Sources 3 and 4 were more general and not strictly necessary for explaining why Arabic voice AI is uniquely difficult.

## Q: What results did ElectroPi's e-commerce case study achieve?

ElectroPi's regional e-commerce retailer case study achieved the following business results:
* 68% deflection of WhatsApp tickets in the first 60 days [1].
* Response time cut from 3 hours to under 8 seconds for automated flows [1].
* CSAT rose from 68% to 89% [1].
* ROI achieved in under 5 months [1].

Citations:
- [1] AI Chatbots for Enterprise: The 2026 Guide | ElectroPi | Case Study: Regional E-Commerce Retailer — Business Results — https://electropi.ai/blog/ai-chatbots-for-enterprise

Judge: faithfulness 5/5 · relevance 5/5 · context precision 5/5 — The system's answer is highly accurate and directly supported by the provided sources. It perfectly answers the question using the exact metrics from the case study results. All retrieved sources were highly relevant to the case study.

## Q: How much does ElectroPi charge for a chatbot project?

Enterprise chatbot projects typically cost fifteen thousand to one hundred fifty thousand dollars, depending on integrations, languages, channels, and model tuning [1].

Citations:
- [1] AI Chatbots for Enterprise: The 2026 Guide | ElectroPi | Frequently Asked Questions — https://electropi.ai/blog/ai-chatbots-for-enterprise

Judge: faithfulness 5/5 · relevance 5/5 · context precision 4/5 — The system's answer is fully faithful to the provided sources, directly quoting the cost range from Source [1]. The answer is highly relevant to the user's question. Context precision is rated 4 because while Source [1] was essential, Sources [2] and [3] were not needed to answer this specific question.

## Q: Who is the CEO of ElectroPi?

I don't have relevant information about that in the ElectroPi knowledge base.

*Refused by guardrail: rerank gate (best 0 < 4)*

Judge: refusal_correct = True — The provided passages do not contain any information about the CEO of ElectroPi, so refusing to answer was correct to avoid hallucination.
