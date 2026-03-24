import json
import base64
import os
from bs4 import BeautifulSoup
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

# ───────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────

def html_to_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(separator=" ").strip()

def save_image(image_id: str, base64_data: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = image_id.replace("/", "_").strip("_") + ".jpg"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(base64_data))
    return filepath

def extract_blocks(block, page_id, chunks, image_dir="images"):
    block_type = block.get("block_type", "")
    html = block.get("html", "")
    images = block.get("images") or {}

    skip_types = {"Page", "ListGroup", "PageFooter", "PageHeader"}
    # Các block này sẽ KHÔNG đệ quy vào children
    leaf_types = {"Table", "Picture", "Figure"}

    if block_type in {"Picture", "Figure"}:
        image_paths = []
        for img_id, img_data in images.items():
            if img_data:
                path = save_image(img_id, img_data, image_dir)
                image_paths.append(path)

        chunks.append({
            "id": block.get("id"),
            "page": page_id,
            "block_type": block_type,
            "text": html_to_text(html) if html else "",
            "image_paths": image_paths,
            "bbox": block.get("bbox"),
            "has_image": True,
            "section": list(block.get("section_hierarchy", {}).values())[0]
                       if block.get("section_hierarchy") else None,
        })
        # ❌ Không đệ quy vào children của ảnh

    elif block_type == "Table":
        # Gộp toàn bộ bảng thành 1 chunk, bỏ qua TableCell riêng lẻ
        chunks.append({
            "id": block.get("id"),
            "page": page_id,
            "block_type": "Table",
            "text": html_to_text(html) if html else "",
            "image_paths": [],
            "bbox": block.get("bbox"),
            "has_image": False,
            "section": list(block.get("section_hierarchy", {}).values())[0]
                       if block.get("section_hierarchy") else None,
        })
        # ❌ Không đệ quy vào TableCell

    elif block_type not in skip_types and html:
        text = html_to_text(html)
        if text:
            chunks.append({
                "id": block.get("id"),
                "page": page_id,
                "block_type": block_type,
                "text": text,
                "image_paths": [],
                "bbox": block.get("bbox"),
                "has_image": False,
                "section": list(block.get("section_hierarchy", {}).values())[0]
                           if block.get("section_hierarchy") else None,
            })

        # ✅ Chỉ đệ quy nếu không phải leaf_types
        if block_type not in leaf_types:
            for child in block.get("children") or []:
                extract_blocks(child, page_id, chunks, image_dir)

    elif block_type in skip_types:
        # Vẫn đệ quy vào Page, ListGroup, ... để lấy children
        for child in block.get("children") or []:
            extract_blocks(child, page_id, chunks, image_dir)

# ───────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────

if __name__ == "__main__":
    PDF_PATH = "ngu-van-9-tap-1_3nhvRcqdNh_c5d39b9109d10a3207cc351533af4258.pdf"
    OUTPUT_JSON = "output.json"
    RAG_OUTPUT = "rag_chunks.json"
    IMAGE_DIR = "images"

    # BƯỚC 1: Convert PDF → JSON
    print("🔄 Đang convert PDF...")
    config = ConfigParser({"output_format": "json"})
    converter = PdfConverter(
        config=config.generate_config_dict(),
        artifact_dict=create_model_dict(),
        processor_list=config.get_processors(),
        renderer=config.get_renderer(),
    )
    rendered = converter(PDF_PATH)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        f.write(rendered.model_dump_json(indent=2))
    print(f"✅ Đã lưu {OUTPUT_JSON}")

    # BƯỚC 2: Parse JSON → RAG chunks
    print("\n🔄 Đang tạo RAG chunks...")
    with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    rag_chunks = []
    for page in data.get("children", []):
        page_id = int(page.get("id", "0/0/0").split("/")[2])
        for child in page.get("children") or []:
            extract_blocks(child, page_id, rag_chunks, IMAGE_DIR)

    # Thêm metadata vào từng chunk
    toc = {
        item["page_id"]: item["title"]
        for item in data.get("metadata", {}).get("table_of_contents", [])
    }
    for chunk in rag_chunks:
        chunk["source"] = PDF_PATH
        chunk["chapter"] = toc.get(chunk["page"], "")

    with open(RAG_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(rag_chunks, f, ensure_ascii=False, indent=2)

    # BƯỚC 3: Thống kê
    text_chunks = [c for c in rag_chunks if not c["has_image"]]
    image_chunks = [c for c in rag_chunks if c["has_image"]]

    print(f"\n✅ Hoàn tất!")
    print(f"   📝 Text chunks : {len(text_chunks)}")
    print(f"   🖼️  Image chunks: {len(image_chunks)} → lưu tại ./{IMAGE_DIR}/")
    print(f"   📦 Tổng        : {len(rag_chunks)} chunks → {RAG_OUTPUT}")

    # Preview 3 chunks đầu
    print("\n── Preview ──")
    for c in rag_chunks[:3]:
        has_img = "🖼️ " if c["has_image"] else "📝"
        print(f"{has_img} [Trang {c['page']} | {c['block_type']}] {c['text'][:80]}")
        if c["image_paths"]:
            print(f"    → Ảnh: {c['image_paths']}")