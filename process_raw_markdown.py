import argparse
from collections import deque
import json
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

_GROQ_DAILY_EXHAUSTED_CODES = frozenset(
    ["tokens_exhausted", "daily_tokens_exceeded", "quota_exceeded"]
)
_MAX_RETRY_WAIT_SECONDS = 120.0


def _load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        print(f"Warning: cannot load {path}: {exc}")


def clean_heading_text(text: str) -> str:
    text = re.sub(r"</?[^>]+>", "", text)
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -\t")


def looks_like_heading_candidate(text: str) -> bool:
    candidate = clean_heading_text(text)
    if len(candidate) < 4 or len(candidate) > 120:
        return False

    words = candidate.split()
    if len(words) > 22:
        return False

    letters = [ch for ch in candidate if ch.isalpha()]
    if not letters:
        return False

    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters)
    return upper_ratio >= 0.6


def fix_headings_with_regex(markdown_text: str) -> str:
    markdown_text = markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = markdown_text.split("\n")
    fixed_lines: list[str] = []

    for line in lines:
        line = re.sub(r"<span id=\"[^\"]+\"></span>", "", line).strip()

        if not line:
            fixed_lines.append("")
            continue

        if line.startswith("!["):
            fixed_lines.append(line)
            continue

        heading_match = re.match(r"^(#{1,6})\s*(.+?)\s*$", line)
        if heading_match:
            marker = heading_match.group(1)
            text = clean_heading_text(heading_match.group(2))
            fixed_lines.append(f"{marker} {text}")
            continue

        bold_match = re.match(r"^\*\*(.+)\*\*$", line)
        if bold_match:
            maybe_heading = clean_heading_text(bold_match.group(1))
            if looks_like_heading_candidate(maybe_heading):
                fixed_lines.append(f"### {maybe_heading}")
                continue

        if looks_like_heading_candidate(line):
            fixed_lines.append(f"## {clean_heading_text(line)}")
        else:
            fixed_lines.append(line)

    fixed = "\n".join(fixed_lines)
    fixed = re.sub(r"\n{3,}", "\n\n", fixed)
    return fixed.strip() + "\n"


def _split_api_keys(raw: str) -> list[str]:
    if not raw.strip():
        return []
    keys: list[str] = []
    for item in re.split(r"[,;\n]", raw):
        key = item.strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def _read_keys_from_file(file_path: str) -> list[str]:
    if not file_path:
        return []
    if not os.path.exists(file_path):
        print(f"Warning: API key file not found: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return _split_api_keys(f.read())
    except Exception as exc:
        print(f"Warning: cannot read API key file {file_path}: {exc}")
        return []


def _mask_key(key: str) -> str:
    if len(key) <= 8:
        return "***"
    return f"{key[:4]}...{key[-4:]}"


def _estimate_tokens(text: str) -> int:
    # Rough estimate to enforce TPM limits without tokenizer dependency.
    return max(1, len(text) // 4)


def _extract_retry_after(exc: Exception) -> float | None:
    try:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None

        for name in [
            "retry-after",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
        ]:
            value = headers.get(name)
            if value is not None:
                wait = float(value)
                if wait >= 0:
                    return wait
    except Exception:
        return None
    return None


def _is_auth_error(msg: str) -> bool:
    return (
        "invalid api key" in msg
        or "incorrect api key" in msg
        or "unauthorized" in msg
        or "authentication" in msg
        or "invalid_api_key" in msg
    )


def _groq_is_daily_limit_error(msg: str) -> bool:
    low = msg.lower()
    return any(code in low for code in _GROQ_DAILY_EXHAUSTED_CODES) or (
        "daily" in low and "quota" in low
    )


def _create_llm_clients(provider: str, attached_keys: list[str] | None = None) -> list[dict[str, Any]]:
    try:
        from openai import OpenAI
    except ImportError:
        print("Warning: skip LLM normalize because openai package is missing")
        return []

    provider = provider.strip().lower()
    attached_keys = attached_keys or []
    keys: list[str] = []

    def append_key(k: str) -> None:
        k = k.strip()
        if k and k not in keys:
            keys.append(k)

    for k in attached_keys:
        append_key(k)

    if provider == "groq":
        for k in _split_api_keys(os.getenv("GROQ_API_KEYS", "")):
            append_key(k)
        append_key(os.getenv("GROQ_API_KEY", ""))

        if not keys:
            print("Warning: skip LLM normalize because GROQ_API_KEY/GROQ_API_KEYS is missing")
            return []

        pool: list[dict[str, Any]] = []
        for key in keys:
            pool.append(
                {
                    "client": OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1"),
                    "label": _mask_key(key),
                    "events": deque(),
                }
            )
        return pool

    for k in _split_api_keys(os.getenv("OPENAI_API_KEYS", "")):
        append_key(k)
    append_key(os.getenv("OPENAI_API_KEY", ""))

    if not keys:
        print("Warning: skip LLM normalize because OPENAI_API_KEY/OPENAI_API_KEYS is missing")
        return []

    pool = []
    for key in keys:
        pool.append(
            {
                "client": OpenAI(api_key=key),
                "label": _mask_key(key),
                "events": deque(),
            }
        )
    return pool


def _prune_events(events: deque[tuple[float, int]], now_ts: float) -> None:
    while events and (now_ts - events[0][0]) >= 60.0:
        events.popleft()


def _usage_in_window(events: deque[tuple[float, int]], now_ts: float) -> tuple[int, int]:
    _prune_events(events, now_ts)
    req_count = len(events)
    token_count = sum(tok for _, tok in events)
    return req_count, token_count


def _can_use_key(
    events: deque[tuple[float, int]],
    now_ts: float,
    req_tokens: int,
    rpm_limit: int,
    tpm_limit: int,
) -> bool:
    used_req, used_tok = _usage_in_window(events, now_ts)
    rpm_ok = rpm_limit <= 0 or (used_req + 1) <= rpm_limit
    tpm_ok = tpm_limit <= 0 or (used_tok + req_tokens) <= tpm_limit
    return rpm_ok and tpm_ok


def _seconds_until_key_available(
    events: deque[tuple[float, int]],
    now_ts: float,
    req_tokens: int,
    rpm_limit: int,
    tpm_limit: int,
) -> float:
    if _can_use_key(events, now_ts, req_tokens, rpm_limit, tpm_limit):
        return 0.0

    waits = sorted({max(0.0, 60.0 - (now_ts - ts)) for ts, _ in events})
    if not waits:
        return 0.25

    for wait_s in waits:
        future = now_ts + wait_s + 0.001
        if _can_use_key(events, future, req_tokens, rpm_limit, tpm_limit):
            return max(wait_s, 0.05)

    return max(waits[-1], 0.25)


def _normalize_single_chunk(
    client,
    model_name: str,
    prompt: str,
    chunk_text: str,
    temperature: float,
) -> tuple[str, int | None]:
    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": chunk_text},
        ],
    )

    content = response.choices[0].message.content or ""
    normalized = content.strip()
    if not normalized:
        raise RuntimeError("empty output")

    used_tokens: int | None = None
    usage = getattr(response, "usage", None)
    if usage is not None:
        used_tokens = getattr(usage, "total_tokens", None)

    return normalized, used_tokens


def _normalize_single_chunk_with_pool(
    clients: list[dict[str, Any]],
    model_name: str,
    prompt: str,
    chunk_text: str,
    provider: str,
    rpm_limit: int,
    tpm_limit: int,
    rr_index: list[int],
    temperature: float,
    max_retries: int,
    retry_base_delay: float,
) -> str:
    if not clients:
        raise RuntimeError("No available API keys")

    provider = provider.strip().lower()
    prompt_tokens = _estimate_tokens(prompt)
    request_tokens = prompt_tokens + (_estimate_tokens(chunk_text) * 2)
    if tpm_limit > 0:
        request_tokens = min(request_tokens, tpm_limit)

    attempt = 0
    while True:
        attempt += 1
        if not clients:
            raise RuntimeError("No available API keys")

        now_ts = time.time()
        chosen_idx = -1
        n = len(clients)
        start = rr_index[0] % n

        for offset in range(n):
            idx = (start + offset) % n
            if _can_use_key(clients[idx]["events"], now_ts, request_tokens, rpm_limit, tpm_limit):
                chosen_idx = idx
                rr_index[0] = idx + 1
                break

        if chosen_idx == -1:
            waits = [
                _seconds_until_key_available(c["events"], now_ts, request_tokens, rpm_limit, tpm_limit)
                for c in clients
            ]
            wait_s = max(0.05, min(waits) if waits else 0.25)
            print(f"  - all keys are throttled, sleeping {wait_s:.2f}s")
            time.sleep(wait_s)
            continue

        selected = clients[chosen_idx]
        key_label = selected["label"]

        try:
            normalized, used_tokens = _normalize_single_chunk(
                selected["client"],
                model_name,
                prompt,
                chunk_text,
                temperature,
            )
            token_to_record = used_tokens if (used_tokens is not None and used_tokens > 0) else request_tokens
            selected["events"].append((time.time(), int(token_to_record)))
            return normalized

        except Exception as exc:
            msg = str(exc).lower()

            if _is_auth_error(msg):
                print(f"Warning: removing invalid key {key_label}")
                clients.pop(chosen_idx)
                rr_index[0] = 0
                continue

            if provider == "groq" and _groq_is_daily_limit_error(msg):
                print(f"Warning: daily quota exhausted for key {key_label}, removing from pool")
                clients.pop(chosen_idx)
                rr_index[0] = 0
                if not clients:
                    raise RuntimeError("All Groq keys hit daily quota") from exc
                continue

            if attempt >= max_retries:
                raise

            wait = _extract_retry_after(exc)
            if wait is None:
                wait = retry_base_delay * (2 ** (attempt - 1))
            wait = max(0.05, min(wait, _MAX_RETRY_WAIT_SECONDS))
            print(f"  - transient error on key {key_label}, retrying in {wait:.2f}s")
            time.sleep(wait)


def _split_big_paragraph(paragraph: str, max_chars: int) -> list[str]:
    paragraph = paragraph.strip()
    if len(paragraph) <= max_chars:
        return [paragraph] if paragraph else []

    parts: list[str] = []
    lines = paragraph.splitlines()
    current = ""

    for line in lines:
        line = line.rstrip()
        candidate = line if not current else f"{current}\n{line}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            parts.append(current)

        if len(line) <= max_chars:
            current = line
            continue

        for i in range(0, len(line), max_chars):
            segment = line[i : i + max_chars]
            if len(segment) == max_chars:
                parts.append(segment)
            else:
                current = segment

    if current:
        parts.append(current)
    return [p for p in parts if p.strip()]


def _split_markdown_for_llm(markdown_text: str, max_chars: int) -> list[str]:
    text = markdown_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(text) <= max_chars:
        return [text]

    sections = re.split(r"(?=^#{1,6}\s+)", text, flags=re.MULTILINE)
    chunks: list[str] = []
    current = ""

    for section in sections:
        section = section.strip()
        if not section:
            continue

        section_parts = [section]
        if len(section) > max_chars:
            section_parts = []
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", section) if p.strip()]
            buffer = ""

            for para in paragraphs:
                candidate = para if not buffer else f"{buffer}\n\n{para}"
                if len(candidate) <= max_chars:
                    buffer = candidate
                    continue

                if buffer:
                    section_parts.append(buffer)

                if len(para) <= max_chars:
                    buffer = para
                else:
                    split_parts = _split_big_paragraph(para, max_chars)
                    if split_parts:
                        section_parts.extend(split_parts[:-1])
                        buffer = split_parts[-1]
                    else:
                        buffer = ""

            if buffer:
                section_parts.append(buffer)

        for part in section_parts:
            candidate = part if not current else f"{current}\n\n{part}"
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                current = part

    if current:
        chunks.append(current.strip())

    return chunks


def _split_in_half(text: str) -> tuple[str, str]:
    text = text.strip()
    if len(text) < 2:
        return text, ""

    mid = len(text) // 2
    left_idx = text.rfind("\n\n", 0, mid)
    right_idx = text.find("\n\n", mid)

    if left_idx == -1 and right_idx == -1:
        cut = mid
    elif left_idx == -1:
        cut = right_idx
    elif right_idx == -1:
        cut = left_idx
    else:
        cut = left_idx if (mid - left_idx) <= (right_idx - mid) else right_idx

    left = text[:cut].strip()
    right = text[cut:].strip()
    if not left or not right:
        return text[:mid].strip(), text[mid:].strip()

    return left, right


def _normalize_chunk_adaptive(
    clients: list[dict[str, Any]],
    model_name: str,
    prompt: str,
    chunk_text: str,
    provider: str,
    min_chunk_chars: int,
    rpm_limit: int,
    tpm_limit: int,
    rr_index: list[int],
    temperature: float,
    max_retries: int,
    retry_base_delay: float,
) -> str:
    try:
        return _normalize_single_chunk_with_pool(
            clients=clients,
            model_name=model_name,
            prompt=prompt,
            chunk_text=chunk_text,
            provider=provider,
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            rr_index=rr_index,
            temperature=temperature,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "reduce the length" in msg and len(chunk_text) > min_chunk_chars:
            left, right = _split_in_half(chunk_text)
            if not left or not right:
                print("Warning: cannot split chunk further, keeping original")
                return chunk_text

            left_norm = _normalize_chunk_adaptive(
                clients,
                model_name,
                prompt,
                left,
                provider,
                min_chunk_chars,
                rpm_limit,
                tpm_limit,
                rr_index,
                temperature,
                max_retries,
                retry_base_delay,
            )
            right_norm = _normalize_chunk_adaptive(
                clients,
                model_name,
                prompt,
                right,
                provider,
                min_chunk_chars,
                rpm_limit,
                tpm_limit,
                rr_index,
                temperature,
                max_retries,
                retry_base_delay,
            )
            return f"{left_norm}\n\n{right_norm}".strip()

        print(f"Warning: skip one chunk due to error: {exc}")
        return chunk_text


# ---------------------------------------------------------------------------
# System prompt: chuẩn hóa Markdown + sửa lỗi tiếng Việt trên heading
# ---------------------------------------------------------------------------

_NORMALIZE_PROMPT = """\
Bạn là công cụ chuẩn hóa cấu trúc Markdown OCR tiếng Việt.
Không thêm hoặc xóa nội dung thực tế.

Quy tắc:
1) Chuẩn hóa cấp độ heading (#, ##, ###, ...).
2) Sửa lỗi tiếng Việt trên tất cả các dòng heading (bắt đầu bằng #):
   - Ghép lại ký tự bị tách rời do lỗi encoding, ví dụ:
       "T r ườ ng" → "Trường"
       "C h ươ ng" → "Chương"
       "Đ ại h ọc" → "Đại học"
   - Sửa dấu thanh điệu bị thiếu hoặc đặt sai vị trí.
   - Chuẩn hóa Unicode về dạng NFC.
3) Giữ nguyên từ ngữ gốc trong phần thân (body) càng nhiều càng tốt.
4) Sửa danh sách Markdown bị lỗi định dạng.
5) Giữ nguyên hình ảnh, công thức và trích dẫn.
6) Chỉ trả về Markdown, không giải thích, không bọc trong code block.\
"""


def normalize_structure_with_llm(
    markdown_text: str,
    model_name: str,
    provider: str,
    llm_max_chars: int,
    llm_min_chunk_chars: int,
    rpm_limit: int,
    tpm_limit: int,
    attached_keys: list[str] | None = None,
    temperature: float = 0.0,
    max_retries: int = 4,
    retry_base_delay: float = 2.0,
) -> str:
    clients = _create_llm_clients(provider, attached_keys=attached_keys)
    if not clients:
        return markdown_text

    if llm_max_chars < 1000:
        llm_max_chars = 1000

    llm_chunks = _split_markdown_for_llm(markdown_text, llm_max_chars)
    if not llm_chunks:
        return markdown_text

    print(
        "LLM normalize: "
        f"{len(llm_chunks)} chunk(s), provider={provider}, model={model_name}, "
        f"keys={len(clients)}, rpm/key={rpm_limit}, tpm/key={tpm_limit}"
    )

    normalized_chunks: list[str] = []
    rr_index = [0]

    for idx, chunk_text in enumerate(llm_chunks, start=1):
        print(f"  - chunk {idx}/{len(llm_chunks)} ({len(chunk_text):,} chars)")
        normalized_chunk = _normalize_chunk_adaptive(
            clients=clients,
            model_name=model_name,
            prompt=_NORMALIZE_PROMPT,
            chunk_text=chunk_text,
            provider=provider,
            min_chunk_chars=llm_min_chunk_chars,
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
            rr_index=rr_index,
            temperature=temperature,
            max_retries=max_retries,
            retry_base_delay=retry_base_delay,
        )
        normalized_chunks.append(normalized_chunk)

    merged = "\n\n".join(x.strip() for x in normalized_chunks if x.strip()).strip()
    return (merged or markdown_text).strip() + "\n"


def _flush_section_buffer(
    chunks: list[dict[str, Any]],
    buffer_lines: list[str],
    heading_path: list[str],
    level: int,
    max_chars: int,
    min_chars: int,
) -> None:
    text = "\n".join(buffer_lines).strip()
    if not text:
        return

    heading = heading_path[-1] if heading_path else "ROOT"
    path = " > ".join(heading_path) if heading_path else "ROOT"

    def push_chunk(content: str) -> None:
        content = content.strip()
        if not content:
            return
        chunks.append(
            {
                "chunk_id": len(chunks) + 1,
                "heading": heading,
                "heading_level": level,
                "heading_path": path,
                "content": content,
                "char_count": len(content),
            }
        )

    if len(text) <= max_chars:
        push_chunk(text)
        return

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current = ""
    for para in paragraphs:
        candidate = para if not current else f"{current}\n\n{para}"
        if len(candidate) <= max_chars or len(current) < min_chars:
            current = candidate
            continue
        push_chunk(current)
        current = para

    push_chunk(current)


def chunk_markdown_by_heading(
    markdown_text: str,
    max_chars: int = 1800,
    min_chars: int = 300,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    heading_path: list[str] = []
    current_level = 0
    buffer_lines: list[str] = []

    for raw_line in markdown_text.splitlines():
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", raw_line.strip())
        if not heading_match:
            buffer_lines.append(raw_line)
            continue

        _flush_section_buffer(
            chunks=chunks,
            buffer_lines=buffer_lines,
            heading_path=heading_path,
            level=current_level,
            max_chars=max_chars,
            min_chars=min_chars,
        )
        buffer_lines = []

        level = len(heading_match.group(1))
        title = clean_heading_text(heading_match.group(2))
        heading_path = heading_path[: level - 1]
        heading_path.append(title)
        current_level = level

    _flush_section_buffer(
        chunks=chunks,
        buffer_lines=buffer_lines,
        heading_path=heading_path,
        level=current_level,
        max_chars=max_chars,
        min_chars=min_chars,
    )

    return chunks


def main() -> None:
    _load_env_file(".env")

    parser = argparse.ArgumentParser(
        description="Process existing raw markdown: regex headings -> optional LLM normalize -> heading chunks"
    )
    parser.add_argument("--input-md", default="output.md")
    parser.add_argument("--fixed-md", default="output.fixed.md")
    parser.add_argument("--normalized-md", default="output.normalized.md")
    parser.add_argument("--chunks-json", default="rag_chunks_by_heading.json")

    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--provider", choices=["openai", "groq"], default=os.getenv("LLM_PROVIDER", "openai"))
    parser.add_argument("--model", default=os.getenv("MARKDOWN_NORMALIZE_MODEL", ""))
    parser.add_argument("--api-key", default="", help="Attach one API key directly")
    parser.add_argument("--api-keys", default="", help="Attach multiple API keys: comma/semicolon/newline")
    parser.add_argument("--api-keys-file", default="", help="File containing API keys (one per line)")

    parser.add_argument("--llm-max-chars", type=int, default=int(os.getenv("LLM_MAX_CHARS", "12000")))
    parser.add_argument("--llm-min-chunk-chars", type=int, default=int(os.getenv("LLM_MIN_CHUNK_CHARS", "1500")))
    parser.add_argument("--rpm-limit", type=int, default=int(os.getenv("LLM_RPM_LIMIT", "45")))
    parser.add_argument("--tpm-limit", type=int, default=int(os.getenv("LLM_TPM_LIMIT", "9000")))
    parser.add_argument("--llm-temperature", type=float, default=float(os.getenv("LLM_TEMPERATURE", "0.0")))
    parser.add_argument("--llm-max-retries", type=int, default=int(os.getenv("LLM_MAX_RETRIES", "4")))
    parser.add_argument("--llm-retry-base-delay", type=float, default=float(os.getenv("LLM_RETRY_BASE_DELAY", "2.0")))

    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--min-chars", type=int, default=300)
    args = parser.parse_args()

    if not os.path.exists(args.input_md):
        raise FileNotFoundError(f"Input markdown not found: {args.input_md}")

    print("Step 1/3: reading raw markdown")
    with open(args.input_md, "r", encoding="utf-8") as f:
        raw_md = f.read()

    print("Step 2/3: regex heading fix")
    fixed_md = fix_headings_with_regex(raw_md)
    with open(args.fixed_md, "w", encoding="utf-8") as f:
        f.write(fixed_md)

    print("Step 3/3: normalize + chunk by heading")
    if args.use_llm:
        attached_keys = _split_api_keys(args.api_keys)
        if args.api_key and args.api_key not in attached_keys:
            attached_keys.append(args.api_key)

        for key in _read_keys_from_file(args.api_keys_file):
            if key not in attached_keys:
                attached_keys.append(key)

        if attached_keys:
            print(f"Attached API keys from CLI/file: {len(attached_keys)} key(s)")

        model_name = args.model or (
            "llama-3.3-70b-versatile" if args.provider == "groq" else "gpt-4.1-mini"
        )

        normalized_md = normalize_structure_with_llm(
            markdown_text=fixed_md,
            model_name=model_name,
            provider=args.provider,
            llm_max_chars=args.llm_max_chars,
            llm_min_chunk_chars=args.llm_min_chunk_chars,
            rpm_limit=args.rpm_limit,
            tpm_limit=args.tpm_limit,
            attached_keys=attached_keys,
            temperature=args.llm_temperature,
            max_retries=args.llm_max_retries,
            retry_base_delay=args.llm_retry_base_delay,
        )
    else:
        normalized_md = fixed_md

    with open(args.normalized_md, "w", encoding="utf-8") as f:
        f.write(normalized_md)

    chunks = chunk_markdown_by_heading(
        normalized_md,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
    )

    with open(args.chunks_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("Done")
    print(f"- Input: {args.input_md} ({len(raw_md):,} chars)")
    print(f"- Fixed markdown: {args.fixed_md} ({len(fixed_md):,} chars)")
    print(f"- Normalized markdown: {args.normalized_md} ({len(normalized_md):,} chars)")
    print(f"- Chunks: {len(chunks):,} -> {args.chunks_json}")


if __name__ == "__main__":
    main()