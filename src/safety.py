import os
import re
import time
from typing import Callable, Optional, Dict, Any
import json


# Configurable safety parameters via environment variables
SAFE_MAX_INPUT_CHARS = int(os.getenv("SAFE_MAX_INPUT_CHARS", "8000"))
SAFE_RATE_LIMIT_PER_MIN = int(os.getenv("SAFE_RATE_LIMIT_PER_MIN", "60"))
SAFE_REQUEST_TIMEOUT_SEC = int(os.getenv("SAFE_REQUEST_TIMEOUT_SEC", "30"))
SAFE_MAX_NEW_TOKENS = int(os.getenv("SAFE_MAX_NEW_TOKENS", "200"))
SAFE_TEMPERATURE_MAX = float(os.getenv("SAFE_TEMPERATURE_MAX", "0.7"))


def clamp_temperature(t: float) -> float:
    try:
        return max(0.0, min(float(t), SAFE_TEMPERATURE_MAX))
    except Exception:
        return 0.2


def sanitize_code_input(code: str) -> str:
    """Sanitize incoming code snippet to mitigate prompt injection and excess size."""
    if code is None:
        return ""
    # Remove common prompt-injection phrases
    injection_patterns = [
        r"(?i)ignore previous instructions",
        r"(?i)disregard all system prompts",
        r"(?i)override safety",
    ]
    cleaned = code
    for pat in injection_patterns:
        cleaned = re.sub(pat, "", cleaned)

    # Strip surrounding backticks to avoid nested code blocks
    cleaned = cleaned.replace("```", "\n")

    # Trim overly long inputs
    if len(cleaned) > SAFE_MAX_INPUT_CHARS:
        cleaned = cleaned[:SAFE_MAX_INPUT_CHARS]
    return cleaned


SECRET_PATTERNS = [
    # Generic API keys and tokens
    r"(?i)\b[A-Za-z0-9_-]{20,}\b",
    # Common formats
    r"(?i)sk-[A-Za-z0-9]{20,}",  # OpenAI-like keys
]


def redact_secrets(text: str) -> str:
    if not text:
        return text
    redacted = text
    for pat in SECRET_PATTERNS:
        redacted = re.sub(pat, "[REDACTED]", redacted)
    return redacted


# Simple PII detection for email and phone numbers
PII_PATTERNS = [
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # emails
    r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}\b",  # phones
]


def redact_pii(text: str) -> str:
    if not text:
        return text
    masked = text
    for pat in PII_PATTERNS:
        masked = re.sub(pat, "[PII REDACTED]", masked)
    return masked


# Minimal content safety blocklist (non-exhaustive, high-level terms)
BLOCKLIST_TERMS = [
    # Violence / harm
    r"(?i)\bkill\b",
    r"(?i)\bharm\b",
    r"(?i)\bviolence\b",
    # Sexual content (explicit)
    r"(?i)\bsexually explicit\b",
    r"(?i)\bporn\b",
    # Hate / harassment (generic terms; no slurs enumerated)
    r"(?i)\bhate speech\b",
    r"(?i)\bharass\w*\b",
]


def violates_policy(text: str) -> bool:
    if not text:
        return False
    for pat in BLOCKLIST_TERMS:
        if re.search(pat, text):
            return True
    return False


def filter_output(text: str) -> str:
    """Apply safety filters to model output and redact sensitive data."""
    if not text:
        return text
    # Redact first, then check policy
    redacted = redact_secrets(redact_pii(text))
    if violates_policy(redacted):
        # Per policy, respond with refusal for harmful content
        return "Sorry, I can't assist with that."
    return redacted.strip()


class RateLimiter:
    """Simple token bucket rate limiter (per-process)."""

    def __init__(self, rate_per_min: int = SAFE_RATE_LIMIT_PER_MIN):
        self.capacity = max(1, int(rate_per_min))
        self.tokens = self.capacity
        self.refill_interval = 60.0
        self.last_refill = time.time()

    def acquire(self) -> bool:
        now = time.time()
        elapsed = now - self.last_refill
        if elapsed >= self.refill_interval:
            refill_count = int(elapsed // self.refill_interval)
            self.tokens = min(self.capacity, self.tokens + refill_count * self.capacity)
            self.last_refill = now
        if self.tokens > 0:
            self.tokens -= 1
            return True
        return False


_GLOBAL_RL = RateLimiter()


def safe_model_call(call: Callable[[], str], timeout_sec: Optional[int] = None) -> str:
    """Guard a model call with rate limiting and timeout.

    If the call fails or times out, returns a safe error message.
    """
    if not _GLOBAL_RL.acquire():
        return "Error: Rate limit exceeded. Please retry later."

    to = timeout_sec if timeout_sec is not None else SAFE_REQUEST_TIMEOUT_SEC

    result_container = {"value": None, "error": None}

    def runner():
        try:
            result_container["value"] = call()
        except Exception as e:
            result_container["error"] = str(e)

    import threading

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(timeout=to)

    if t.is_alive():
        return "Error: Request timed out."
    if result_container["error"]:
        return f"Error: {result_container['error']}"
    return filter_output(result_container["value"] or "")


def audit_log(event: str, data: Dict[str, Any]) -> None:
    """Append a JSON line to the experiments/logs/safety.log file with redaction."""
    try:
        sanitized: Dict[str, Any] = {}
        for k, v in (data or {}).items():
            if isinstance(v, str):
                sanitized[k] = redact_secrets(redact_pii(v))
            else:
                sanitized[k] = v
        sanitized["event"] = event
        sanitized["ts"] = int(time.time())
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        log_dir = os.path.join(root, "experiments", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "safety.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sanitized, ensure_ascii=False) + "\n")
    except Exception:
        # Best-effort logging; do not raise
        pass
