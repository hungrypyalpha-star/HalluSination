"""
One-shot pipeline for the MT hallucination task of HalluSinatioin:

  1. Reads sinhala_machine_translation_dataset.csv
  2. Sends every English source sentence to the LLM for translation
     (concurrent, retry-with-backoff, rate-limited, crash-safe resume)
  3. Scores every successful translation against the gold reference with:
       - COMET       (neural embedding metric, src+mt+ref)
       - GEMBA-MQM   (LLM-as-judge, MQM severity-weighted error spans)
       - Chrf

Dataset columns used (input CSV):
    id, domain, source_text_en, translation_sin, source,
    model_name, model_output_sin, hallucination_type, metric_label,
    annotator_id, notes

Organized/output schema (json + csv):
    id | domain | source_text_en | translation_sin | source |
    model_name | model_output_sin | hallucination_type | metric_label |
    annotator_id | notes | comet_score | gemba_mqm_score | gemba_mqm_errors

Usage:
    python hallusination_mt.py

Optional environment variables:
    OPENAI_BASE_URL       API base URL                        (default: http://127.0.0.1:8317/v1)
    OPENAI_API_KEY        API key                             (default: built-in local proxy key)
    OPENAI_MODEL          Translation model name              (default: claude-sonnet-4-6)
    NUM_SAMPLES           Rows to translate                   (default: all rows)
    MAX_WORKERS           Parallel threads (translation)      (default: 5)
    REQUESTS_PER_SECOND   Request-rate ceiling (translation)  (default: 2.0)
    RESUME                "0" disables resume                 (default: enabled)
    INPUT_FILE            Path to source CSV
    OUTPUT_DIR            Directory for all output files      (default: script dir)
    OUTPUT_BASENAME       Base filename stem                  (default: hallucination_mt_<model>)

    COMPUTE_COMET         "0" skips COMET scoring              (default: enabled)
    COMET_MODEL           COMET checkpoint name                (default: Unbabel/wmt22-comet-da)
    COMET_BATCH_SIZE      Batch size for COMET inference       (default: 16)

    COMPUTE_GEMBA         "0" skips GEMBA-MQM scoring           (default: enabled)
    GEMBA_MODEL            Judge model for GEMBA-MQM            (default: gpt-5.5)
    GEMBA_BASE_URL         Judge API base URL                   (default: same as OPENAI_BASE_URL)
    GEMBA_API_KEY          Judge API key                        (default: same as OPENAI_API_KEY)
    GEMBA_MAX_WORKERS      Parallel threads (judge calls)       (default: MAX_WORKERS)
    GEMBA_REQUESTS_PER_SECOND  Request-rate ceiling (judge)     (default: REQUESTS_PER_SECOND)

    FORCE_RECOMPUTE_METRICS  "1" recomputes COMET/GEMBA even if already present (default: 0)
"""

import csv
import json
import os
import random
import re
import sys
import threading
import time
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

MODEL_NAME          = os.environ.get("OPENAI_MODEL", "gpt-5.5")
NUM_SAMPLES         = int(os.environ.get("NUM_SAMPLES", "310"))
MAX_WORKERS         = int(os.environ.get("MAX_WORKERS", "5"))
REQUESTS_PER_SECOND = float(os.environ.get("REQUESTS_PER_SECOND", "2.0"))
RESUME              = os.environ.get("RESUME", "1") != "0"

COMPUTE_COMET     = os.environ.get("COMPUTE_COMET", "1") != "0"
COMET_MODEL       = os.environ.get("COMET_MODEL", "Unbabel/wmt22-comet-da")
COMET_BATCH_SIZE  = int(os.environ.get("COMET_BATCH_SIZE", "16"))

COMPUTE_GEMBA             = os.environ.get("COMPUTE_GEMBA", "1") != "0"
GEMBA_MODEL               = os.environ.get("GEMBA_MODEL", "gemini-3.6-flash-high")
GEMBA_BASE_URL            = os.environ.get("GEMBA_BASE_URL", API_BASE_URL)
GEMBA_API_KEY             = os.environ.get("GEMBA_API_KEY", API_KEY)
GEMBA_MAX_WORKERS         = int(os.environ.get("GEMBA_MAX_WORKERS", str(MAX_WORKERS)))
GEMBA_REQUESTS_PER_SECOND = float(os.environ.get("GEMBA_REQUESTS_PER_SECOND", str(REQUESTS_PER_SECOND)))

FORCE_RECOMPUTE_METRICS = os.environ.get("FORCE_RECOMPUTE_METRICS", "0") == "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
INPUT_FILE = os.environ.get(
    "INPUT_FILE",
    os.path.join(PROJECT_ROOT, "datasets", "sinhala_machine_translation_dataset.csv"),
)
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", os.path.join(PROJECT_ROOT, "claude_results", "mt", MODEL_NAME))
OUTPUT_BASENAME = os.environ.get("OUTPUT_BASENAME", f"hallucination_mt_{MODEL_NAME}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

RAW_JSON_PATH       = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_raw.json")
ORGANIZED_JSON_PATH = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_organized.json")
ORGANIZED_CSV_PATH  = os.path.join(OUTPUT_DIR, f"{OUTPUT_BASENAME}_organized.csv")

MAX_RETRIES    = 3
RETRY_BASE_SEC = 2.0  # doubles each retry, plus jitter

CSV_COLUMNS = [
    "id", "domain", "source_text_en", "translation_sin", "source",
    "model_name", "model_output_sin", "hallucination_type",
    "comet_score", "gemba_mqm_score", "gemba_mqm_errors",
    "annotator_id", "notes"
]

SYSTEM_PROMPT = """\
1) ඔබේ කාර්යය නම් ලබා දෙන ඉංග්‍රීසි ඡේදය හෝ වාක්‍යය නිවැරදිව සිංහලට පරිවර්තනය කිරීමයි.
2) පරිවර්තනය ලබා දීමේදී සරල පෙළ පමණක් භාවිත කරන්න.
3) ප්‍රතිචාරයේ සිංහල පරිවර්තනය පමණක් ලබා දෙන්න.
"""

# MQM-style severity weights used to turn GEMBA-MQM error spans into a
# single 0-100 score. These mirror the weighting scheme described in
# Kocmi & Federmann (GEMBA-MQM): minor errors cost little, major errors
# cost more, critical/hallucination-level errors cost the most.
GEMBA_SEVERITY_WEIGHTS = {"minor": 1, "major": 5, "critical": 25}

GEMBA_MQM_SYSTEM_PROMPT = """\
You are an expert translation-quality annotator applying the Multidimensional \
Quality Metrics (MQM) framework to evaluate English-to-Sinhala machine translation, \
with a focus on hallucination detection (unsupported additions, omissions, \
detachment from the source, wrong-language output, repetition).

You will be given:
  - SOURCE: the original English sentence/paragraph
  - REFERENCE: a gold-standard human Sinhala translation (for guidance only)
  - CANDIDATE: a Sinhala translation produced by an MT/LLM system, to be evaluated

Judge the CANDIDATE strictly against the SOURCE meaning (the REFERENCE is a helpful \
aid, not the only acceptable phrasing). Identify every quality error and classify \
each by severity:
  - "critical": the candidate is unrelated to the source, fabricates a large amount \
    of unsupported content, is in the wrong language, or is otherwise unusable
  - "major": a significant unsupported addition, omission, mistranslation, or \
    contradiction that changes the meaning
  - "minor": a small fluency, style, or terminology issue that does not change \
    the meaning

Respond with ONLY a single valid JSON object and nothing else (no markdown fences, \
no commentary), in exactly this shape:
{"errors": [{"severity": "major", "category": "addition|omission|mistranslation|contradiction|wrong_language|repetition|other", "explanation": "short explanation in English"}], "score": <integer 0-100>}

Compute "score" yourself as 100 minus the MQM-weighted penalty (minor=-1, major=-5, \
critical=-25 per error), floored at 0. If there are no errors, return \
{"errors": [], "score": 100}.
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


translate_rate_limiter = RateLimiter(REQUESTS_PER_SECOND)
gemba_rate_limiter = RateLimiter(GEMBA_REQUESTS_PER_SECOND)


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
            if not str(r.get("model_translation_sin", "")).startswith("[ERROR]")
        }
    except Exception:
        return {}


def is_translated(r: dict) -> bool:
    return bool(r.get("model_translation_sin")) and not str(
        r.get("model_translation_sin", "")
    ).startswith("[ERROR]")


# Stage 1 helpers — translation

def build_translate_prompt(source_en: str) -> str:
    return (
        "පිළිතුරු සිංහලෙන් පමණක් දෙන්න. පහත ඉංග්‍රීසි ඡේදය හෝ වාක්‍යය සිංහලට පරිවර්තනය කරන්න:\n\n"
        + source_en.strip()
        + "\n\nසිංහල පරිවර්තනය:"
    )


def call_translate_llm(client, source_en: str) -> dict:
    """Rate-limited translation call with exponential backoff + jitter."""
    user_msg = build_translate_prompt(source_en)
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        translate_rate_limiter.acquire()
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=10000,
            )
            elapsed = round(time.time() - t0, 2)
            choice = response.choices[0]
            usage = response.usage
            return {
                "translation": choice.message.content.strip(),
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


# Stage 2 helpers — GEMBA-MQM (LLM-as-judge)

def build_gemba_prompt(source_en: str, reference_sin: str, candidate_sin: str) -> str:
    return (
        f"SOURCE:\n{source_en}\n\n"
        f"REFERENCE:\n{reference_sin}\n\n"
        f"CANDIDATE:\n{candidate_sin}\n\n"
        "JSON:"
    )


def parse_gemba_response(text: str) -> dict:
    """Parse the judge's JSON reply; tolerate markdown fences / stray text."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        parsed = json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse GEMBA-MQM JSON from: {text[:200]}")
        parsed = json.loads(match.group(0))

    errors = parsed.get("errors", []) or []
    penalty = sum(GEMBA_SEVERITY_WEIGHTS.get(str(e.get("severity", "")).lower(), 0) for e in errors)
    computed_score = max(0, 100 - penalty)
    model_score = parsed.get("score")
    score = model_score if isinstance(model_score, (int, float)) else computed_score

    return {"errors": errors, "score": score, "computed_score": computed_score}


def call_gemba_llm(judge_client, source_en: str, reference_sin: str, candidate_sin: str) -> dict:
    user_msg = build_gemba_prompt(source_en, reference_sin, candidate_sin)
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        gemba_rate_limiter.acquire()
        try:
            response = judge_client.chat.completions.create(
                model=GEMBA_MODEL,
                messages=[
                    {"role": "system", "content": GEMBA_MQM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw_text = response.choices[0].message.content.strip()
            parsed = parse_gemba_response(raw_text)
            parsed["raw_response"] = raw_text
            return parsed
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                wait = RETRY_BASE_SEC * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                time.sleep(wait)

    raise last_exc


def needs_gemba(r: dict) -> bool:
    if not is_translated(r):
        return False
    if not r.get("reference_sin") or not r.get("source_text_en"):
        return False
    if FORCE_RECOMPUTE_METRICS:
        return True
    return r.get("gemba_mqm_score") in (None, "")


# Stage 3 helpers — COMET

def needs_comet(r: dict) -> bool:
    if not is_translated(r):
        return False
    if not r.get("reference_sin") or not r.get("source_text_en"):
        return False
    if FORCE_RECOMPUTE_METRICS:
        return True
    return r.get("comet_score") in (None, "")


def compute_comet_scores(all_results: list):
    if not COMPUTE_COMET:
        print("COMET: skipped (COMPUTE_COMET=0)")
        return

    targets = [r for r in all_results if needs_comet(r)]
    if not targets:
        print("COMET: nothing to score (already computed or no eligible rows).")
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

    print(f"COMET: scoring {len(targets)} rows using {COMET_MODEL} ...")
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
        {"src": r["source_text_en"], "mt": r["model_translation_sin"], "ref": r["reference_sin"]}
        for r in targets
    ]

    # Score + write incrementally in chunks
    chunk_size = max(1, COMET_BATCH_SIZE * 4)
    for start in range(0, len(targets), chunk_size):
        chunk_targets = targets[start:start + chunk_size]
        chunk_data = data[start:start + chunk_size]
        output = model.predict(chunk_data, batch_size=COMET_BATCH_SIZE, gpus=gpus)
        for r, s in zip(chunk_targets, output.scores):
            r["comet_score"] = round(float(s), 4)
        write_all_outputs(all_results)
        print(f"  COMET progress: {min(start + chunk_size, len(targets))}/{len(targets)}")

    print("COMET: done.")


# GEMBA-MQM worker

def process_gemba_row(judge_client, r: dict, all_results: list, stats: dict, print_lock):
    row_id = r.get("id", "")
    try:
        result = call_gemba_llm(judge_client, r["source_text_en"], r["reference_sin"], r["model_translation_sin"])
        r["gemba_mqm_score"] = result["score"]
        r["gemba_mqm_error_list"] = result["errors"]
        r["gemba_mqm_raw_response"] = result["raw_response"]
        status_str = f"OK  score={result['score']}  errors={len(result['errors'])}"
        with stats["lock"]:
            stats["success"] += 1
    except Exception as exc:
        r["gemba_mqm_score"] = ""
        r["gemba_mqm_error_list"] = []
        r["gemba_mqm_raw_response"] = f"[ERROR] {exc}"
        status_str = f"ERROR: {exc}"
        with stats["lock"]:
            stats["errors"] += 1

    with stats["lock"]:
        stats["done"] += 1
        done, total = stats["done"], stats["total"]

    write_all_outputs(all_results)

    with print_lock:
        print(f"[GEMBA {done:>3}/{total}] ID#{row_id} | {status_str}")


def compute_gemba_scores(judge_client, all_results: list):
    if not COMPUTE_GEMBA:
        print("GEMBA-MQM: skipped (COMPUTE_GEMBA=0)")
        return

    targets = [r for r in all_results if needs_gemba(r)]
    if not targets:
        print("GEMBA-MQM: nothing to score (already computed or no eligible rows).")
        return

    print(f"GEMBA-MQM: scoring {len(targets)} rows using judge model {GEMBA_MODEL} ...")
    stats = {"success": 0, "errors": 0, "done": 0, "total": len(targets), "lock": threading.Lock()}
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=GEMBA_MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_gemba_row, judge_client, r, all_results, stats, print_lock)
            for r in targets
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                with print_lock:
                    print(f"  [CRITICAL] Unhandled error in GEMBA-MQM worker: {exc}")

    print(f"GEMBA-MQM: done. success={stats['success']} errors={stats['errors']}")


# Transform: raw prediction record -> organized target schema record

def summarize_gemba_errors(error_list) -> str:
    if not error_list:
        return ""
    parts = []
    for e in error_list:
        sev = e.get("severity", "?")
        cat = e.get("category", "?")
        expl = e.get("explanation", "")
        parts.append(f"[{sev}/{cat}] {expl}")
    return " | ".join(parts)


def to_organized(raw: dict) -> dict:
    return {
        "id":                 raw.get("id", ""),
        "domain":             raw.get("domain", ""),
        "source_text_en":     raw.get("source_text_en", ""),
        "translation_sin":    raw.get("reference_sin", ""),
        "source":             raw.get("source_url", ""),
        "model_name":         raw.get("model_name", ""),
        "model_output_sin":   raw.get("model_translation_sin", ""),
        "hallucination_type": raw.get("human_hallucination_type", ""),
        "annotator_id":       raw.get("annotator_id", ""),
        "notes":              raw.get("notes", ""),
        "comet_score":        raw.get("comet_score", ""),
        "gemba_mqm_score":    raw.get("gemba_mqm_score", ""),
        "gemba_mqm_errors":   summarize_gemba_errors(raw.get("gemba_mqm_error_list", [])),
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
            if not str(r.get("model_translation_sin", "")).startswith("[ERROR]")
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



# Translation worker: process one row

def process_translate_row(client, idx, total, row, all_results, stats, print_lock):
    row_id           = safe(row, "id")
    domain           = safe(row, "domain")
    source_en        = safe(row, "source_text_en")
    reference_sin    = safe(row, "translation_sin")
    source_url       = safe(row, "source")
    model_name_csv   = safe(row, "model_name")
    model_output_csv = safe(row, "model_output_sin")
    halluc_type_csv  = safe(row, "hallucination_type")
    annotator_id     = safe(row, "annotator_id")
    notes            = safe(row, "notes")

    if not source_en:
        with print_lock:
            print(f"[{idx:>3}/{total}] ID#{row_id} -- SKIPPED (empty source_text_en)")
        return

    base = {
        "sample_index":             idx,
        "id":                       row_id,
        "domain":                   domain,
        "source_text_en":           source_en,
        "reference_sin":            reference_sin,
        "source_url":               source_url,
        "csv_model_name":           model_name_csv,
        "csv_model_output_sin":     model_output_csv,
        "human_hallucination_type": halluc_type_csv,
        "annotator_id":             annotator_id,
        "notes":                    notes,
        "model_name":               MODEL_NAME,
        "label_hallucination_type": "",
        "label_metric":             "",
        "label_notes":              "",
        "comet_score":              "",
        "gemba_mqm_score":          "",
        "gemba_mqm_error_list":     [],
        "gemba_mqm_raw_response":   "",
    }

    try:
        result = call_translate_llm(client, source_en)
        status_str = f"OK  ({result['latency_seconds']}s | {result['total_tokens']} tok)"
        record = {
            **base,
            "model_translation_sin": result["translation"],
            "finish_reason":         result["finish_reason"],
            "prompt_tokens":         result["prompt_tokens"],
            "completion_tokens":     result["completion_tokens"],
            "total_tokens":          result["total_tokens"],
            "latency_seconds":       result["latency_seconds"],
            "timestamp":             datetime.now(timezone.utc).isoformat(),
        }
        with stats["lock"]:
            stats["success"] += 1
            stats["tokens"]  += result["total_tokens"]
            stats["done"]    += 1
    except Exception as exc:
        status_str = f"ERROR: {exc}"
        record = {
            **base,
            "model_translation_sin": f"[ERROR] {exc}",
            "finish_reason":         "error",
            "prompt_tokens":         0,
            "completion_tokens":     0,
            "total_tokens":          0,
            "latency_seconds":       0,
            "timestamp":             datetime.now(timezone.utc).isoformat(),
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
        preview = source_en[:80].replace("\n", " ")
        print(
            f"[{idx:>3}/{total}] ID#{row_id} | {domain}\n"
            f"  EN : {preview}{'...' if len(source_en) > 80 else ''}\n"
            f"  {status_str}\n"
            f"  elapsed={elapsed_total:.0f}s  ETA={eta_sec:.0f}s  "
            f"tok/s={tok_per_s:.1f}  done={done}/{total}\n"
        )


def run_translation_stage(client, all_results, existing):
    print("\nLoading source CSV ...")
    all_samples = load_csv(INPUT_FILE)
    samples = all_samples[:NUM_SAMPLES]
    print(f"Loaded rows: {len(samples)} / {len(all_samples)}")

    pending = []
    skipped = 0
    for idx, row in enumerate(samples, start=1):
        row_id = safe(row, "id")
        if RESUME and row_id in existing:
            skipped += 1
        else:
            pending.append((idx, row))

    print(f"  Already translated (resume): {skipped}")
    print(f"  To translate now            : {len(pending)}\n")

    if not pending:
        print("Translation stage: everything already done.")
        return

    stats = {
        "success": 0, "errors": 0, "tokens": 0, "done": 0,
        "lock": threading.Lock(), "start_time": time.time(),
    }
    print_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_translate_row, client, idx, len(pending), row, all_results, stats, print_lock): idx
            for idx, row in pending
        }
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                with print_lock:
                    print(f"  [CRITICAL] Unhandled error in translation worker: {exc}")

    elapsed_total = time.time() - stats["start_time"]
    print(
        f"\nTranslation stage done in {elapsed_total:.1f}s — "
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
    judge_client = (
        client
        if (GEMBA_BASE_URL == API_BASE_URL and GEMBA_API_KEY == API_KEY)
        else OpenAI(base_url=GEMBA_BASE_URL, api_key=GEMBA_API_KEY)
    )

    print("=" * 70)
    print("  MT Hallucination Pipeline — translate + COMET + GEMBA-MQM + organize")
    print("=" * 70)
    print(f"  Translation model : {MODEL_NAME}")
    print(f"  Translation API   : {API_BASE_URL}")
    print(f"  Judge model (GEMBA): {GEMBA_MODEL}")
    print(f"  Judge API         : {GEMBA_BASE_URL}")
    print(f"  Translate threads : {MAX_WORKERS}  (RPS={REQUESTS_PER_SECOND})")
    print(f"  GEMBA threads     : {GEMBA_MAX_WORKERS}  (RPS={GEMBA_REQUESTS_PER_SECOND})")
    print(f"  COMET             : {'enabled (' + COMET_MODEL + ')' if COMPUTE_COMET else 'disabled'}")
    print(f"  GEMBA-MQM         : {'enabled' if COMPUTE_GEMBA else 'disabled'}")
    print(f"  Resume            : {'enabled' if RESUME else 'disabled'}")
    print(f"  Input file        : {INPUT_FILE}")
    print(f"  Raw json          : {RAW_JSON_PATH}")
    print(f"  Organized json    : {ORGANIZED_JSON_PATH}")
    print(f"  Organized csv     : {ORGANIZED_CSV_PATH}")
    print("=" * 70)

    existing = load_existing_raw(RAW_JSON_PATH) if RESUME else {}
    all_results = list(existing.values())

    # Stage 1: translations
    run_translation_stage(client, all_results, existing)
    write_all_outputs(all_results)

    # Stage 2: COMET (reference-based neural metric, batched)
    compute_comet_scores(all_results)

    # Stage 3: GEMBA-MQM (LLM-as-judge, concurrent + rate-limited like translation)
    compute_gemba_scores(judge_client, all_results)

    write_all_outputs(all_results)

    # Final summary
    successes = [r for r in all_results if is_translated(r)]
    errors = [r for r in all_results if not is_translated(r)]
    comet_scored = [r for r in successes if r.get("comet_score") not in (None, "")]
    gemba_scored = [r for r in successes if r.get("gemba_mqm_score") not in (None, "")]

    print("=" * 70)
    print(f"Raw json        -> {RAW_JSON_PATH}")
    print(f"Organized json  -> {ORGANIZED_JSON_PATH}")
    print(f"Organized csv   -> {ORGANIZED_CSV_PATH}")
    print("\n=== SUMMARY ===")
    print(f"  Translated (success) : {len(successes)}")
    print(f"  Translation errors   : {len(errors)}")
    print(f"  COMET scored         : {len(comet_scored)}")
    print(f"  GEMBA-MQM scored     : {len(gemba_scored)}")
    if comet_scored:
        avg_comet = sum(r["comet_score"] for r in comet_scored) / len(comet_scored)
        print(f"  Avg COMET score      : {avg_comet:.4f}")
    if gemba_scored:
        avg_gemba = sum(r["gemba_mqm_score"] for r in gemba_scored) / len(gemba_scored)
        print(f"  Avg GEMBA-MQM score  : {avg_gemba:.1f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
