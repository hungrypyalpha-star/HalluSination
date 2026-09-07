# HalluSination: An Empirical Study of Hallucination for Sinhala Across Multiple Tasks and Domains

## Overview

Hallucination — model output that is fluent but unsupported by its source or context — is well studied for high-resource languages but largely unaddressed for Sinhala. **HalluSination** is the first empirical source-grounded hallucination study for Sinhala, covering two tasks:

* **Machine Translation (MT)** — English → Sinhala translation across 4 domains.
* **Question Answering (QA)** — Source-grounded reading comprehension, including refusal on unanswerable questions.

The study evaluates **14 Large Language Models (LLMs)** alongside **3 conventional MT systems** (Google Translate, NLLB, MADLAD-400) on ~600 curated items. Native Sinhala speakers manually annotated over 9,000 model outputs using a task-specific hallucination taxonomy.

## Motivation

Existing Sinhala NLP resources (FLORES, SinhalaMMLU, Sinhala-GLUE) evaluate general translation quality or language understanding, but none explicitly measure whether a model's output is grounded in its source. HalluSination fills this gap by characterizing fine-grained hallucination patterns and refusal behaviors in low-resource settings.

## Benchmark Composition

| Track | Items | Domain / Source Breakdown | Notes |
| --- | --- | --- | --- |
| **Machine Translation** | 310 | News, Legal, Cultural, General | English → Sinhala pairs |
| **Question Answering** | 302 | Short-answer (226), Unanswerable (76) | Source-grounded Q&A + refusal cases |
| **Total** | **612** | | **9,000+ total human annotations** |

### Machine Translation (310 items)

| Domain | Count | Primary Data Source |
| --- | --- | --- |
| **General** | 75 | FLORES / FLORES-200 |
| **News** | 76 | FLORES / FLORES-200 |
| **Legal** | 74 | Sri Lankan Constitution, Penal Code, Acts |
| **Cultural** | 75 | Global-PIQA (Sinhala non-parallel subset) |

### Question Answering (302 items)

| Category | Count | Primary Source | Purpose |
| --- | --- | --- | --- |
| **Short-answer (SQuAD-style)** | 150 | SiQuAD | Extractive reading comprehension |
| **Short-answer (Multiple Choice)** | 77 | Belebele (Sinhala) | Adapted to short-answer form |
| **Unanswerable** | 76 | FLORES (SQuAD 2.0 design) | Refusal resistance & hallucination testing |

## Dataset Schema

### Machine Translation (`sinhala_machine_translation_dataset.csv`)

| Field | Description |
| --- | --- |
| `id` | Unique identifier |
| `domain` | `general` / `legal` / `cultural` / `news` |
| `source_text_en` | English source sentence or paragraph |
| `translation_sin` | Gold Sinhala translation reference |
| `source` | Origin URL / dataset reference |
| `model_name` | Evaluated model or baseline |
| `model_output_sin` | Generated Sinhala translation |
| `hallucination_type` | Primary label: `Faithful`, `Omission`, `Mistranslation`, `Hallucination` |
| `hallucination_subtype` | Subtype if hallucination: `extraneous`, `repetition`, `detached`, `partial`, `incorrect_target_language`, `other` |
| `annotator_id` | Identifier of native Sinhala annotator |
| `notes` | Comments / justification |
| `comet_score` | COMET neural quality score |
| `gemba_mqm_score` | GEMBA-MQM LLM-judge score (0–100) |
| `gemba_mqm_errors` | Severity-tagged error spans from GEMBA-MQM |

### Question Answering (`sinhala_Q_A_dataset.csv`)

| Field | Description |
| --- | --- |
| `id` | Unique identifier |
| `source_passage_sin` | Sinhala passage used as source evidence |
| `question_sin` | Sinhala question posed to the model |
| `reference_answer_sin` | Gold Sinhala answer or `UNANSWERABLE` |
| `model_name` | Evaluated model |
| `model_output_sin` | Generated Sinhala answer |
| `hallucination_type` | `Supported`, `Unsupported`, `Correct refusal`, `False refusal` |
| `annotator_id` | Identifier of native Sinhala annotator |
| `notes` | Comments / justification |
| `exact_match` | Exact match binary score |
| `token_f1` | Token F1 metric score |
| `chrf` | Character n-gram F-score (chrF / chrF3) |

## Annotation Taxonomy

### Machine Translation Taxonomy

Each translation receives exactly **one primary label**:

* **Faithful** — Meaning preserved without introducing unsupported content.
* **Omission** — Important source content missing.
* **Mistranslation** — Content is present and grounded in source, but rendered incorrectly (e.g., incorrect word form, spelling, inflection, transliterations of standard words).
* **Hallucination** — Introduces content unsupported by source lacking meaningful semantic connection.

If labeled **Hallucination**, one of six subcategories is assigned:

* **Extraneous hallucination** — Fabricates semantically significant unsupported content.
* **Repetition** — Looping or oscillatory text pattern.
* **Detached hallucination** — Fluent text largely unrelated to source meaning.
* **Partial hallucination** — Portion is unrelated to source while remainder is grounded.
* **Incorrect target language / untranslated content** — Text is in wrong language or copied untranslated.
* **Other** — Custom hallucination requiring annotator justification.

### Question Answering Taxonomy

Evaluated using the AIS source-verification framework:

* **Supported** — Every factual claim is directly grounded in or reasonably inferred from the source passage.
* **Unsupported** — Contains claims unverified or contradicted by the source passage.
* **Correct refusal** — Model appropriately abstains when evidence is insufficient.
* **False refusal** — Model declines to answer an answerable question.

## Evaluated Models & Results Summary

### Evaluated Systems

* **Closed-Source LLMs:** Claude Opus 4.6, Claude Sonnet 4.6, Gemini-3.1-Pro, Gemini-3.6-Flash, GPT-5.5.
* **Open-Weight LLMs:** DeepSeek-V4-Pro-0813, DeepSeek-V4-Flash-0731, Llama-4-Maverick, Llama-3.1-70B-Instruct, Llama-3.1-8B-Instruct, Qwen3.5-27B, Qwen3.5-9B, Qwen3.5-4B, Mistral-Large-2512.
* **MT Baselines:** Google Translate, NLLB, MADLAD-400.

### Key Performance Summary (QA Track)

| Model | Human Acc. (%) | Supported (%) | Unsupported (%) | False Ref. (%) | Correct Ref. (%) | Token F1 | NoAns-EM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Claude-Opus-4.6** | **97.0** | 97.3 | 0.4 | 2.2 | 96.1 | 36.8 | 82.9 |
| **Gemini-3.1-Pro** | **96.4** | 95.6 | 1.8 | 2.7 | 98.7 | 54.6 | 98.7 |
| **Claude-Sonnet-4.6** | **95.7** | 95.6 | 1.8 | 2.7 | 96.1 | 42.1 | 92.1 |
| **Gemini-3.6-Flash** | **93.4** | 91.2 | 7.5 | 1.3 | **100.0** | 54.1 | 100.0 |
| **DeepSeek-V4-Pro** | **93.0** | 92.5 | 2.7 | 4.9 | 94.7 | 65.4 | 94.7 |
| **GPT-5.5** | **92.7** | 94.7 | 4.9 | 0.4 | 86.8 | 56.2 | 85.5 |
| **DeepSeek-V4-Flash** | **91.7** | 90.3 | 6.6 | 3.1 | 96.1 | 60.1 | 94.7 |
| **Llama-4-Maverick** | **90.1** | 96.0 | 0.9 | 3.1 | 72.4 | 58.0 | 72.4 |
| **Qwen3.5-27B** | **85.8** | 81.0 | 16.4 | 2.7 | **100.0** | 48.2 | 100.0 |
| **Qwen3.5-9B** | **83.1** | 83.2 | 8.8 | 8.0 | 82.9 | 56.6 | 67.1 |
| **Qwen3.5-4B** | **67.2** | 63.3 | 12.8 | 23.9 | 78.9 | 37.0 | 64.5 |
| **Mistral-Large** | **56.3** | 68.1 | 5.8 | 26.1 | 21.1 | 25.4 | 5.3 |
| **Llama-3.1-70B** | **51.7** | 58.4 | 40.3 | 0.0 | 31.6 | 43.8 | 31.6 |
| **Llama-3.1-8B** | **38.7** | 50.9 | 38.1 | 0.4 | 2.6 | 16.9 | 0.0 |