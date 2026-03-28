import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser

# ───────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────

PDF_PATH = "Kết nối tập 1.pdf"
OUTPUT_MD = "output.md"
IMAGE_DIR = "images"

# ───────────────────────────────────────────
# CONVERT PDF → MARKDOWN
# ───────────────────────────────────────────

print("🔄 Đang convert PDF sang Markdown...")

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

rendered = converter(PDF_PATH)

# Lưu markdown
with open(OUTPUT_MD, "w", encoding="utf-8") as f:
    f.write(rendered.markdown)

# Lưu ảnh (nếu có)
# Lưu ảnh (nếu có)
if rendered.images:
    os.makedirs(IMAGE_DIR, exist_ok=True)
    for img_name, img_data in rendered.images.items():
        img_path = os.path.join(IMAGE_DIR, img_name)
        img_data.save(img_path)  # PIL Image.save() trực tiếp
    print(f"   🖼️  Đã lưu {len(rendered.images)} ảnh → ./{IMAGE_DIR}/")

print(f"✅ Hoàn tất! Markdown đã lưu tại: {OUTPUT_MD}")
print(f"   📄 Độ dài: {len(rendered.markdown):,} ký tự")