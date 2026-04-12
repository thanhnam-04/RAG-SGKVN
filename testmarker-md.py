import argparse
import json
import os
import re
from typing import Any


def convert_pdf_to_markdown(pdf_path: str, image_dir: str) -> tuple[str, int]:
    """Convert PDF to markdown with Marker and optionally save extracted images."""
    try:
        from marker.config.parser import ConfigParser  # type: ignore[import-not-found]
        from marker.converters.pdf import PdfConverter  # type: ignore[import-not-found]
        from marker.models import create_model_dict  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Marker chưa được cài. Hãy cài marker-pdf để dùng bước convert PDF."
        ) from exc

    config = ConfigParser({
        "output_format": "markdown",
        "output_dir": ".",
    })

    converter = PdfConverter(
        config=config.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config.get_processors(),
        renderer=config.get_renderer(),
    )

    rendered = converter(pdf_path)
    image_count = 0

    if rendered.images:
        os.makedirs(image_dir, exist_ok=True)
        for img_name, img_data in rendered.images.items():
            img_path = os.path.join(image_dir, img_name)
            img_data.save(img_path)
            image_count += 1

    return rendered.markdown, image_count


def clean_heading_text(text: str) -> str:
    text = re.sub(r"</?[^>]+>", "", text)
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -\t")


def looks_like_heading_candidate(text: str) -> bool:
    """Heuristic: detect OCR lines that likely represent headings."""
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
    """Basic heading cleanup before sending text to LLM."""
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


def _create_llm_client(provider: str):
    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️  Bỏ qua bước LLM: chưa cài openai")
        return None

    normalized_provider = provider.strip().lower()
    if normalized_provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            print("⚠️  Bỏ qua bước LLM: chưa có GROQ_API_KEY")
            return None
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("⚠️  Bỏ qua bước LLM: chưa có OPENAI_API_KEY")
        return None
    return OpenAI(api_key=api_key)


def normalize_structure_with_llm(markdown_text: str, model_name: str, provider: str) -> str:
    """Optional LLM step to normalize markdown structure while preserving content."""
    client = _create_llm_client(provider)
    if client is None:
        return markdown_text

    prompt = (
        "Bạn là chuyên gia chuẩn hóa markdown OCR. "
        "Nhiệm vụ: chỉ chỉnh cấu trúc markdown, KHÔNG thêm/xóa nội dung factual.\n"
        "Yêu cầu:\n"
        "1) Chuẩn hóa heading level hợp lý (#, ##, ###).\n"
        "2) Giữ nguyên văn bản gốc nhiều nhất có thể.\n"
        "3) Sửa list markdown cơ bản nếu bị lỗi định dạng.\n"
        "4) Giữ nguyên công thức, trích dẫn, hình ảnh markdown.\n"
        "5) Trả về duy nhất markdown đã chuẩn hóa."
    )

    try:
        response = client.responses.create(
            model=model_name,
            temperature=0,
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": markdown_text},
            ],
        )
        normalized = (response.output_text or "").strip()
        if not normalized:
            print("⚠️  Bỏ qua kết quả LLM: phản hồi rỗng")
            return markdown_text
        return normalized + "\n"
    except Exception as exc:
        print(f"⚠️  Bỏ qua bước LLM do lỗi: {exc}")
        return markdown_text


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
        if level <= 0:
            level = 1

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
    parser = argparse.ArgumentParser(
        description="PDF -> raw markdown -> regex fix -> optional LLM normalize -> chunk by heading"
    )
    parser.add_argument("--pdf-path", default="Lịch sử Việt Nam 5.pdf")
    parser.add_argument("--input-md", default="", help="Skip Marker step and read markdown from file")
    parser.add_argument("--raw-md", default="output.md")
    parser.add_argument("--fixed-md", default="output.fixed.md")
    parser.add_argument("--normalized-md", default="output.normalized.md")
    parser.add_argument("--chunks-json", default="rag_chunks_by_heading.json")
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--provider", choices=["openai", "groq"], default=os.getenv("LLM_PROVIDER", "openai"))
    parser.add_argument("--model", default=os.getenv("MARKDOWN_NORMALIZE_MODEL", ""))
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--min-chars", type=int, default=300)
    args = parser.parse_args()

    if args.input_md:
        print("🔄 BƯỚC 1: Đọc markdown có sẵn")
        with open(args.input_md, "r", encoding="utf-8") as f:
            raw_md = f.read()
        image_count = 0
    else:
        print("🔄 BƯỚC 1: Marker convert PDF -> markdown thô")
        raw_md, image_count = convert_pdf_to_markdown(args.pdf_path, args.image_dir)

    with open(args.raw_md, "w", encoding="utf-8") as f:
        f.write(raw_md)

    print("🔧 BƯỚC 2: Regex fix heading cơ bản")
    fixed_md = fix_headings_with_regex(raw_md)
    with open(args.fixed_md, "w", encoding="utf-8") as f:
        f.write(fixed_md)

    print("🤖 BƯỚC 3: Normalize structure")
    if args.use_llm:
        model_name = args.model or (
            "llama-3.3-70b-versatile" if args.provider == "groq" else "gpt-4.1-mini"
        )
        normalized_md = normalize_structure_with_llm(fixed_md, model_name, args.provider)
    else:
        print("ℹ️  Bỏ qua LLM (chạy với --use-llm để bật)")
        normalized_md = fixed_md

    with open(args.normalized_md, "w", encoding="utf-8") as f:
        f.write(normalized_md)

    print("🧩 BƯỚC 4: Chunk theo heading")
    chunks = chunk_markdown_by_heading(
        normalized_md,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
    )
    with open(args.chunks_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("\n✅ Hoàn tất pipeline")
    if args.input_md:
        print(f"   📄 Input markdown: {args.input_md}")
    else:
        print(f"   📄 PDF: {args.pdf_path}")
        print(f"   🖼️  Images saved: {image_count} -> ./{args.image_dir}/")
    print(f"   📝 Raw markdown: {args.raw_md} ({len(raw_md):,} chars)")
    print(f"   🛠️  Regex fixed: {args.fixed_md} ({len(fixed_md):,} chars)")
    print(f"   ✨ Normalized: {args.normalized_md} ({len(normalized_md):,} chars)")
    print(f"   📦 Chunks: {len(chunks):,} -> {args.chunks_json}")


if __name__ == "__main__":
    main()