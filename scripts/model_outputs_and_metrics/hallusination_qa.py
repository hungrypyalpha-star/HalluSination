"""
One-shot pipeline for the Question Answering hallucination task.

  1. Reads sinhala_Q_A_dataset.csv and classifies every row into one of
     three sub-types, based on the gold answer:
       - unanswerable  -> reference_answer_sin starts with the refusal
                          token (default "UNANSWERABLE", matching gold spelling)
       - short_answer  -> otherwise

  2. Sends every passage+question to the LLM for an answer (concurrent,
     retry-with-backoff, rate-limited, crash-safe resume). The model is
     instructed to reply with the exact refusal token when the passage
     doesn't support an answer.

  3. Scores every prediction with the metric appropriate to its sub-type:
       - short_answer  -> Exact Match + token-level F1 vs. gold answer
       - unanswerable  -> SQuAD v2 NoAns EM (did the model output the
                          exact refusal token?)

  4. On every completed unit of work, rewrites THREE output files so you
     always have an up-to-date, consistent snapshot:
       - <basename>_raw.json        full audit trail (all fields, resume key)
       - <basename>_organized.json  clean target schema (Table 6 in proposal)
       - <basename>_organized.csv   same schema, ready for annotators

Dataset columns used (input CSV):
    link, question_number, source_passage_sin, question_sin,
    reference_answer_sin, model_name, model_output_sin,
    hallucination_type, metric_value, annotator_id, notes

Organized/output schema (json + csv):
    id | source_passage_sin | question_sin | reference_answer_sin |
    source | model_name | model_output_sin | hallucination_type |
    annotator_id | notes | qa_subtype | exact_match |
     token_f1 | comet_score | squad_v2_noans_em | chrf | chrf3

Usage:
    python run_qa_hallucination_pipeline.py

Optional environment variables:
    OPENAI_BASE_URL       Proxy/API base URL                  (default: http://127.0.0.1:8317/v1)
    OPENAI_API_KEY        Proxy/API key                       (default: built-in local proxy key)
    OPENAI_MODEL          Model name                          (default: claude-sonnet-4-6)
    NUM_SAMPLES           Rows to answer                      (default: all rows)
    MAX_WORKERS           Parallel threads                    (default: 5)
    REQUESTS_PER_SECOND   Request-rate ceiling                (default: 2.0)
    RESUME                "0" disables resume                 (default: enabled)
    INPUT_FILE            Path to source CSV
    OUTPUT_DIR            Directory for all output files      (default: script dir)
    OUTPUT_BASENAME       Base filename stem                  (default: hallucination_qa_<model>)

    SHORT_ANSWER_CHAR_THRESHOLD  Gold-answer char length cutoff between
                                  short- and long-answer sub-types  (default: 150)
    REFUSAL_TOKEN                 Exact refusal token, must match gold
                                  data spelling                     (default: UNANSWERABLE)
    FORCE_RECOMPUTE_METRICS  "1" recomputes all metrics even if already present (default: 0)
"""

import csv
import json
import os
import random
import re
import string
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

os.environ.setdefault("UV_LINK_MODE", "copy")
sys.stdout.reconfigure(encoding="utf-8")

# Load .env (if present) to populate os.environ before reading config
def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
    if not os.path.exists(env_path):
        return
    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                k, v = line.split('=', 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k not in os.environ:
                    os.environ[k] = v
_load_env_file()


# Configuration
API_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
API_KEY      = os.environ.get("OPENAI_API_KEY", "")

MODEL_NAME          = os.environ.get("OPENAI_MODEL", "gemini-3.6-flash-high")
NUM_SAMPLES         = int(os.environ.get("NUM_SAMPLES", "350"))
MAX_WORKERS         = int(os.environ.get("MAX_WORKERS", "5"))
REQUESTS_PER_SECOND = float(os.environ.get("REQUESTS_PER_SECOND", "2.0"))
RESUME              = os.environ.get("RESUME", "1") != "0"

SHORT_ANSWER_CHAR_THRESHOLD = int(os.environ.get("SHORT_ANSWER_CHAR_THRESHOLD", "150"))
REFUSAL_TOKEN                = os.environ.get("REFUSAL_TOKEN", "UNANSWERABLE")

FORCE_RECOMPUTE_METRICS = os.environ.get("FORCE_RECOMPUTE_METRICS", "0") == "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
INPUT_FILE = os.environ.get(
    "INPUT_FILE",
    os.path.join(PROJECT_ROOT, "datasets", "sinhala_Q&A_dataset.csv"),
)
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "implicit_qa", "qa", MODEL_NAME))
OUTPUT_BASENAME = os.environ.get("OUTPUT_BASENAME", f"hallucination_qa_{MODEL_NAME}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_JSON_PATH       = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_raw.json")
ORGANIZED_JSON_PATH = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_organized.json")
ORGANIZED_CSV_PATH  = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_organized.csv")

MAX_RETRIES    = 3
RETRY_BASE_SEC = 2.0  # doubles each retry, plus jitter

CSV_COLUMNS = [
    "id", "source_passage_sin", "question_sin", "reference_answer_sin",
    "source", "model_name", "model_output_sin", "hallucination_type",
    "qa_subtype", "exact_match", "token_f1", "comet_score", "squad_v2_noans_em",
    "chrf", "chrf3", "annotator_id", "notes",
]

SYSTEM_PROMPT = f"""\
1) සන්දර්භ‍යේ තිබෙන තොරතුරු පමණක් පදනම් කරගෙන, ප්‍රශ්නයට පිළිතුරු සිංහල බසින් සපයන්න.
2) පිළිතුරු සැපයීමේදී සරල පෙළ පමණක් භාවිත කරන්න.
"""

# Rate limiter — smooths concurrent requests to a global RPS ceiling
class RateLimiter:
    """Thread-safe token-bucket limiter shared by all worker threads."""

    def __init__(self, rate_per_sec: float, burst: int = None):
        self.rate = max(rate_per_sec, 0.01)
        self.capacity = burst if burst is not None else max(1, int(self.rate))
        self.tokens = float(self.capacity)
        self.lock = threading.Lock()
        self.last = time.monotonic()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                elapsed = now - self.last
                self.last = now
                self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait = (1.0 - self.tokens) / self.rate
            time.sleep(wait)


qa_rate_limiter = RateLimiter(REQUESTS_PER_SECOND)


# Generic helpers
def safe(row: dict, key: str) -> str:
    return (row.get(key) or "").strip()


def load_csv(filepath: str) -> list:
    with open(filepath, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_existing_raw(path: str) -> dict:
    """Resume support: keyed by row id, only successful (non-error) rows."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {
            str(r.get("id", "")): r
            for r in data
            if not str(r.get("model_output_sin", "")).startswith("[ERROR]")
        }
    except Exception:
        return {}


def is_answered(r: dict) -> bool:
    return bool(r.get("model_output_sin")) and not str(
        r.get("model_output_sin", "")
    ).startswith("[ERROR]")



# Sub-type classification
def classify_subtype(reference_answer: str) -> str:
    ref = (reference_answer or "").strip()
    if ref.upper().startswith(REFUSAL_TOKEN.upper()):
        return "unanswerable"
    if len(ref) <= SHORT_ANSWER_CHAR_THRESHOLD:
        return "short_answer"
    return "long_answer"



# Stage 1 helpers — answer generation
def build_qa_prompt(context_sin: str, question_sin: str) -> str:
    return (
        "සන්දර්භය:\n" + context_sin
        + "\n\nප්‍රශ්නය:\n" + question_sin
        + "\n\nපිළිතුර:\n"
    )


def call_qa_llm(client, context_sin: str, question_sin: str) -> dict:
    """Rate-limited QA call with exponential backoff + jitter."""
    user_msg = build_qa_prompt(context_sin, question_sin)
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        qa_rate_limiter.acquire()
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=4096,
            )
            elapsed = round(time.time() - t0, 2)
            choice = response.choices[0]
            usage = response.usage
            return {
                "prediction": choice.message.content.strip(),
                "finish_reason": choice.finish_reason,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "latency_seconds": elapsed,
            }
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = RETRY_BASE_SEC * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                time.sleep(wait)

    raise last_exc



# Stage 2 helpers — metrics
_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}\u0964\u0965]")


def normalize_answer(text: str) -> str:
    text = (text or "").strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_em_f1(prediction: str, gold: str) -> tuple:
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(gold)

    em = 1.0 if norm_pred == norm_gold else 0.0

    pred_tokens = norm_pred.split()
    gold_tokens = norm_gold.split()
    if not pred_tokens or not gold_tokens:
        f1 = 1.0 if pred_tokens == gold_tokens else 0.0
        return em, f1

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return em, 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1


def _char_ngrams(text: str, n: int) -> Counter:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) < n:
        return Counter()
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def compute_chrf(prediction: str, gold: str, char_order: int = 6, beta: float = 2.0) -> float:
    if not prediction and not gold:
        return 100.0
    if not prediction or not gold:
        return 0.0

    precisions = []
    recalls = []
    for n in range(1, char_order + 1):
        pred_ngrams = _char_ngrams(prediction, n)
        gold_ngrams = _char_ngrams(gold, n)
        overlap = sum((pred_ngrams & gold_ngrams).values())
        precisions.append(overlap / sum(pred_ngrams.values()) if pred_ngrams else 0.0)
        recalls.append(overlap / sum(gold_ngrams.values()) if gold_ngrams else 0.0)

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall) * 100


def is_refusal(prediction: str, token: str) -> bool:
    norm_pred = normalize_answer(prediction)
    norm_token = normalize_answer(token)
    return norm_pred == norm_token


def needs_short_metrics(r: dict) -> bool:
    if r.get("qa_subtype") != "short_answer" or not is_answered(r):
        return False
    if FORCE_RECOMPUTE_METRICS:
        return True
    return r.get("exact_match") in (None, "")


def needs_unanswerable_metrics(r: dict) -> bool:
    if r.get("qa_subtype") != "unanswerable" or not is_answered(r):
        return False
    if FORCE_RECOMPUTE_METRICS:
        return True
    return r.get("squad_v2_noans_em") in (None, "")


def needs_comet(r: dict) -> bool:
    if r.get("qa_subtype") != "long_answer" or not is_answered(r):
        return False
    if not r.get("reference_answer_sin") or not r.get("source_passage_sin"):
        return False
    if FORCE_RECOMPUTE_METRICS:
        return True
    return r.get("comet_score") in (None, "")


def needs_chrf(r: dict) -> bool:
    if not is_answered(r) or not r.get("reference_answer_sin"):
        return False
    if FORCE_RECOMPUTE_METRICS:
        return True
    return r.get("chrf") in (None, "") or r.get("chrf3") in (None, "")


def compute_short_answer_metrics(all_results: list):
    targets = [r for r in all_results if needs_short_metrics(r)]
    if not targets:
        print("EM/F1 (short-answer): nothing to score.")
        return
    print(f"EM/F1 (short-answer): scoring {len(targets)} rows ...")
    for r in targets:
        em, f1 = compute_em_f1(r["model_output_sin"], r["reference_answer_sin"])
        r["exact_match"] = em
        r["token_f1"] = round(f1, 4)
    write_all_outputs(all_results)
    avg_em = sum(r["exact_match"] for r in targets) / len(targets)
    avg_f1 = sum(r["token_f1"] for r in targets) / len(targets)
    print(f"  done. avg EM={avg_em:.3f}  avg F1={avg_f1:.3f}")


def compute_unanswerable_metrics(all_results: list):
    targets = [r for r in all_results if needs_unanswerable_metrics(r)]
    if not targets:
        print("SQuAD v2 NoAns EM (unanswerable): nothing to score.")
        return
    print(f"SQuAD v2 NoAns EM (unanswerable): scoring {len(targets)} rows ...")
    for r in targets:
        r["squad_v2_noans_em"] = 1.0 if is_refusal(r["model_output_sin"], REFUSAL_TOKEN) else 0.0
    write_all_outputs(all_results)
    avg = sum(r["squad_v2_noans_em"] for r in targets) / len(targets)
    print(f"  done. avg NoAns EM={avg:.3f} (correct-refusal rate)")


def compute_chrf_scores(all_results: list):
    targets = [r for r in all_results if needs_chrf(r)]
    if not targets:
        print("ChrF/ChrF3: nothing to score.")
        return
    print(f"ChrF/ChrF3: scoring {len(targets)} rows ...")
    for r in targets:
        r["chrf"] = round(compute_chrf(r["model_output_sin"], r["reference_answer_sin"], char_order=6), 4)
        r["chrf3"] = round(compute_chrf(r["model_output_sin"], r["reference_answer_sin"], char_order=3), 4)
    write_all_outputs(all_results)
    avg_chrf = sum(r["chrf"] for r in targets) / len(targets)
    avg_chrf3 = sum(r["chrf3"] for r in targets) / len(targets)
    print(f"  done. avg ChrF={avg_chrf:.3f}  avg ChrF3={avg_chrf3:.3f}")


def compute_comet_scores(all_results: list):
    if not COMPUTE_COMET:
        print("COMET (long-answer): skipped (COMPUTE_COMET=0)")
        return

    targets = [r for r in all_results if needs_comet(r)]
    if not targets:
        print("COMET (long-answer): nothing to score (already computed or no eligible rows).")
        return

    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print(
            "COMET: 'unbabel-comet' not installed — skipping COMET scoring.\n"
            "  Install with: pip install unbabel-comet --break-system-packages\n"
            "  (first run also downloads the checkpoint from Hugging Face; "
            "you may need `huggingface-cli login` for gated checkpoints such as "
            f"{COMET_MODEL})"
        )
        return

    print(f"COMET (long-answer): scoring {len(targets)} rows using {COMET_MODEL} ...")
    try:
        model_path = download_model(COMET_MODEL)
        model = load_from_checkpoint(model_path)
    except Exception as exc:
        print(f"COMET: failed to load model ({exc}) — skipping COMET scoring.")
        return

    try:
        import torch
        gpus = 1 if torch.cuda.is_available() else 0
    except ImportError:
        gpus = 0

    data = [
        {"src": r["source_passage_sin"], "mt": r["model_output_sin"], "ref": r["reference_answer_sin"]}
        for r in targets
    ]

    chunk_size = max(1, COMET_BATCH_SIZE * 4)
    for start in range(0, len(targets), chunk_size):
        chunk_targets = targets[start:start + chunk_size]
        chunk_data = data[start:start + chunk_size]
        output = model.predict(chunk_data, batch_size=COMET_BATCH_SIZE, gpus=gpus)
        for r, s in zip(chunk_targets, output.scores):
            r["comet_score"] = round(float(s), 4)
        write_all_outputs(all_results)
        print(f"  COMET progress: {min(start + chunk_size, len(targets))}/{len(targets)}")

    print("COMET (long-answer): done.")



# Transform: raw prediction record -> organized target schema record
def to_organized(raw: dict) -> dict:
    return {
        "id":                    raw.get("id", ""),
        "source":                raw.get("source", ""),  # populated from CSV 'source' column
        "source_passage_sin":    raw.get("source_passage_sin", ""),
        "question_sin":          raw.get("question_sin", ""),
        "reference_answer_sin":  raw.get("reference_answer_sin", ""),
        "model_name":            raw.get("model_name", ""),
        "model_output_sin":      raw.get("model_output_sin", ""),
        "hallucination_type":    raw.get("human_hallucination_type", ""),
        "chrf":                  raw.get("chrf", ""),
        "chrf3":                 raw.get("chrf3", ""),
        "annotator_id":          raw.get("annotator_id", ""),
        "notes":                 raw.get("notes", ""),
        "qa_subtype":            raw.get("qa_subtype", ""),
        "exact_match":           raw.get("exact_match", ""),
        "token_f1":              raw.get("token_f1", ""),
        "comet_score":           raw.get("comet_score", ""),
        "squad_v2_noans_em":     raw.get("squad_v2_noans_em", ""),
    }


def sort_key(rec):
    try:
        return int(rec["id"])
    except (ValueError, TypeError):
        return rec["id"]



# Thread-safe incremental writer — rewrites all three artifacts atomically
_write_lock = threading.Lock()


def write_all_outputs(all_raw_results: list):
    """Atomically (re)write raw json, organized json, and organized csv."""
    with _write_lock:
        raw_sorted = sorted(all_raw_results, key=lambda r: r.get("sample_index", 0))

        tmp = RAW_JSON_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(raw_sorted, f, ensure_ascii=False, indent=2)
        os.replace(tmp, RAW_JSON_PATH)

        organized = [
            to_organized(r) for r in raw_sorted
            if not str(r.get("model_output_sin", "")).startswith("[ERROR]")
        ]
        organized.sort(key=sort_key)

        tmp = ORGANIZED_JSON_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(organized, f, ensure_ascii=False, indent=2)
        os.replace(tmp, ORGANIZED_JSON_PATH)

        tmp = ORGANIZED_CSV_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
            writer.writeheader()
            writer.writerows(organized)
        os.replace(tmp, ORGANIZED_CSV_PATH)



# Answer-generation worker: process one row

def process_qa_row(client, idx, total, row, all_results, stats, print_lock):
    row_id           = safe(row, "id") or str(idx)
    source           = safe(row, "source")
    context_sin      = safe(row, "source_passage_sin")
    question_sin     = safe(row, "question_sin")
    reference_answer = safe(row, "reference_answer_sin")
    model_name_csv   = safe(row, "model_name")
    model_output_csv = safe(row, "model_output_sin")
    halluc_type_csv  = safe(row, "hallucination_type")
    annotator_id     = safe(row, "annotator_id")
    notes            = safe(row, "notes")

    subtype = classify_subtype(reference_answer)

    if not context_sin or not question_sin:
        with print_lock:
            print(f"[{idx:>3}/{total}] ID#{row_id} -- SKIPPED (empty passage/question)")
        return

    base = {
        "sample_index":             idx,
        "id":                       row_id,
        "source":                   source,
        "source_passage_sin":       context_sin,
        "question_sin":             question_sin,
        "reference_answer_sin":     reference_answer,
        "csv_model_name":           model_name_csv,
        "csv_model_output_sin":     model_output_csv,
        "human_hallucination_type": halluc_type_csv,
        "annotator_id":             annotator_id,
        "notes":                    notes,
        "model_name":               MODEL_NAME,
        "qa_subtype":               subtype,
        "exact_match":              "",
        "token_f1":                 "",
        "comet_score":              "",
        "squad_v2_noans_em":        "",
        "chrf":                     "",
        "chrf3":                    "",
    }

    try:
        result = call_qa_llm(client, context_sin, question_sin)
        status_str = f"OK  ({result['latency_seconds']}s | {result['total_tokens']} tok)"
        record = {
            **base,
            "model_output_sin":  result["prediction"],
            "finish_reason":     result["finish_reason"],
            "prompt_tokens":     result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "total_tokens":      result["total_tokens"],
            "latency_seconds":   result["latency_seconds"],
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        }
        with stats["lock"]:
            stats["success"] += 1
            stats["tokens"]  += result["total_tokens"]
            stats["done"]    += 1
    except Exception as exc:
        status_str = f"ERROR: {exc}"
        record = {
            **base,
            "model_output_sin":  f"[ERROR] {exc}",
            "finish_reason":     "error",
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "total_tokens":      0,
            "latency_seconds":   0,
            "timestamp":         datetime.now(timezone.utc).isoformat(),
        }
        with stats["lock"]:
            stats["errors"] += 1
            stats["done"]   += 1

    with _write_lock:
        all_results.append(record)
    write_all_outputs(all_results)

    elapsed_total = time.time() - stats["start_time"]
    done = stats["done"]
    remaining = total - done
    eta_sec = (elapsed_total / done * remaining) if done else 0
    tok_per_s = stats["tokens"] / elapsed_total if elapsed_total > 0 else 0

    with print_lock:
        q_preview = question_sin[:80].replace("\n", " ")
        print(
            f"[{idx:>3}/{total}] ID#{row_id} | {subtype}\n"
            f"  Q  : {q_preview}{'...' if len(question_sin) > 80 else ''}\n"
            f"  {status_str}\n"
            f"  elapsed={elapsed_total:.0f}s  ETA={eta_sec:.0f}s  "
            f"tok/s={tok_per_s:.1f}  done={done}/{total}\n"
        )


def run_qa_stage(client, all_results, existing):
    print("\nLoading source CSV ...")
    all_samples = load_csv(INPUT_FILE)
    samples = all_samples[:NUM_SAMPLES]
    print(f"Loaded rows: {len(samples)} / {len(all_samples)}")

    pending = []
    skipped = 0
    for idx, row in enumerate(samples, start=1):
        row_id = safe(row, "id") or str(idx)
        if RESUME and row_id in existing:
            skipped += 1
        else:
            pending.append((idx, row))

    print(f"  Already answered (resume): {skipped}")
    print(f"  To answer now             : {len(pending)}\n")

    if not pending:
        print("QA answer-generation stage: everything already done.")
        return

    stats = {
        "success": 0, "errors": 0, "tokens": 0, "done": 0,
        "lock": threading.Lock(), "start_time": time.time(),
    }
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_qa_row, client, idx, len(pending), row, all_results, stats, print_lock): idx
            for idx, row in pending
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                with print_lock:
                    print(f"  [CRITICAL] Unhandled error in QA worker: {exc}")

    elapsed_total = time.time() - stats["start_time"]
    print(
        f"\nQA answer-generation stage done in {elapsed_total:.1f}s — "
        f"success={stats['success']} errors={stats['errors']} tokens={stats['tokens']:,}"
    )



# Main

def main():
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed.\n  Run: pip install openai --break-system-packages")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("=" * 70)
    print("  QA Hallucination Pipeline — answer + EM/F1 + COMET + SQuADv2 + organize")
    print("=" * 70)
    print(f"  Model                : {MODEL_NAME}")
    print(f"  API                  : {API_BASE_URL}")
    print(f"  Threads              : {MAX_WORKERS}  (RPS={REQUESTS_PER_SECOND})")
    print(f"  Short/long threshold : {SHORT_ANSWER_CHAR_THRESHOLD} chars")
    print(f"  Refusal token        : {REFUSAL_TOKEN}")
    print(f"  COMET (long-answer)  : {'enabled (' + COMET_MODEL + ')' if COMPUTE_COMET else 'disabled'}")
    print(f"  Resume               : {'enabled' if RESUME else 'disabled'}")
    print(f"  Input file           : {INPUT_FILE}")
    print(f"  Raw json             : {RAW_JSON_PATH}")
    print(f"  Organized json       : {ORGANIZED_JSON_PATH}")
    print(f"  Organized csv        : {ORGANIZED_CSV_PATH}")
    print("=" * 70)

    existing = load_existing_raw(RAW_JSON_PATH) if RESUME else {}
    all_results = list(existing.values())

    # Stage 1: answer generation
    run_qa_stage(client, all_results, existing)
    write_all_outputs(all_results)

    # Stage 2: sub-type-specific metrics
    compute_short_answer_metrics(all_results)   # EM + token F1
    compute_unanswerable_metrics(all_results)   # SQuAD v2 NoAns EM
    compute_chrf_scores(all_results)            # ChrF + ChrF3
    compute_comet_scores(all_results)           # COMET (long-answer)

    write_all_outputs(all_results)

    # Final summary
    answered = [r for r in all_results if is_answered(r)]
    errors = [r for r in all_results if not is_answered(r)]
    by_subtype = {"short_answer": [], "long_answer": [], "unanswerable": []}
    for r in answered:
        by_subtype.setdefault(r.get("qa_subtype", ""), []).append(r)

    print("=" * 70)
    print(f"Raw json        -> {RAW_JSON_PATH}")
    print(f"Organized json  -> {ORGANIZED_JSON_PATH}")
    print(f"Organized csv   -> {ORGANIZED_CSV_PATH}")
    print("\n=== SUMMARY ===")
    print(f"  Answered (success) : {len(answered)}")
    print(f"  Errors             : {len(errors)}")

    short_rows = by_subtype.get("short_answer", [])
    scored_short = [r for r in short_rows if r.get("exact_match") not in (None, "")]
    if scored_short:
        avg_em = sum(r["exact_match"] for r in scored_short) / len(scored_short)
        avg_f1 = sum(r["token_f1"] for r in scored_short) / len(scored_short)
        print(f"  Short-answer  (n={len(short_rows):>3}): EM={avg_em:.3f}  F1={avg_f1:.3f}")

    scored_chrf = [r for r in answered if r.get("chrf") not in (None, "")]
    if scored_chrf:
        avg_chrf = sum(r["chrf"] for r in scored_chrf) / len(scored_chrf)
        avg_chrf3 = sum(r["chrf3"] for r in scored_chrf) / len(scored_chrf)
        print(f"  ChrF          (n={len(scored_chrf):>3}): ChrF={avg_chrf:.3f}  ChrF3={avg_chrf3:.3f}")

    unans_rows = by_subtype.get("unanswerable", [])
    scored_unans = [r for r in unans_rows if r.get("squad_v2_noans_em") not in (None, "")]
    if scored_unans:
        avg_noans = sum(r["squad_v2_noans_em"] for r in scored_unans) / len(scored_unans)
        print(f"  Unanswerable  (n={len(unans_rows):>3}): SQuADv2 NoAns EM={avg_noans:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
