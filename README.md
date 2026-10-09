# HalluSination: A Multi-Task, Multi-Domain Sinhala Hallucination Benchmark

## Overview

**HalluSination** is the first source-grounded hallucination benchmark for Sinhala. It covers two tasks:

- **Machine Translation (MT)** — English → Sinhala across 4 domains
- **Question Answering (QA)** — hallucination and faithfulness evaluation on source-grounded QA, including refusal behavior on unanswerable questions

17 systems (14 LLMs + 3 conventional MT baselines) were evaluated on ~600 curated items, with native Sinhala speakers manually annotating 9,000+ model outputs using task-specific hallucination taxonomies.

## Why

Existing Sinhala resources (FLORES, SinhalaMMLU, Sinhala-GLUE, etc.) measure translation quality or general language understanding, but none directly measure whether outputs are grounded in their source. This benchmark fills that gap.

## Benchmark

| Track | Items | Domains |
|---|---|---|
| Machine Translation | 306 | News, Legal, Cultural, General |
| Question Answering | 302 | Short-answer (226), Unanswerable (76) |
| **Total** | **~600** | **9,000+ human annotations** |

- **MT sources:** FLORES/FLORES-200 (General, News), Sri Lankan Constitution/Penal Code/Acts (Legal), Global-PIQA Sinhala subset (Cultural)
- **QA sources:** SiQuAD (short-answer), Belebele-Sinhala (adapted MC → short-answer), manually built unanswerable items over FLORES passages

## Annotation Taxonomies

**MT** — one primary label per output: `Faithful`, `Omission`, `Mistranslation`, `Degenerated Repetitive Output`, `Hallucination` (with subtypes: extraneous, repetition, detached, partial, incorrect target language/untranslated, other).

**QA** — one label per output, via an AIS-style source-verification test: `Supported`, `Unsupported`, `Correct refusal`, `False refusal`, `Degenerated Repetitive Output`.

Inter-annotator agreement: MT primary labels κ = 0.59 (72.4% raw agreement); QA labels κ = 0.90 (92.0% three-way agreement).

## Systems Evaluated

- **Closed-source LLMs:** 2 Claude models, 2 Gemini models, 1 GPT model
- **Open-weight LLMs:** DeepSeek (2 sizes), Llama (3 variants), Qwen (3 sizes), Mistral
- **MT baselines:** Google Translate, NLLB, MADLAD-400

## Headline Findings

**QA:**
- Human-judged accuracy ranges from 38.7% to 97.0% across models; ten of fourteen models score ≥83%.
- Correct-refusal behavior on unanswerable questions ranges from 2.6% to 100%.
- Automatic overlap metrics (EM, Token F1) correlate only moderately with human judgments and penalize longer, still-grounded answers — one top model ranks 9th by Token F1 but 1st by human Supported rate.

**MT:**
- Human-judged Faithful rates range from 0.7% to 92.5%; a handful of closed-source systems exceed 84% faithful, while eight systems fall below 20%.
- Hallucination is concentrated among open-weight models (up to 28.4%); closed-source systems stay ≤2%.
- Legal content has the lowest faithful rate overall, driven mainly by mistranslation, not hallucination.
- COMET and GEMBA-MQM correlate strongly with human faithfulness at the system level (ρ ≈ 0.83–0.95) but miss domain-specific and sentence-level hallucination patterns.

## Dataset Files (schema summary)

- `datasets/sinhala_machine_translation_dataset.csv` — source/reference translation/model output, hallucination type & subtype, annotator ID, COMET/GEMBA-MQM scores
- `datasets/sinhala_Q_A_dataset.csv` — passage/question/reference answer/model output, hallucination type, annotator ID, EM/Token F1/chrF
