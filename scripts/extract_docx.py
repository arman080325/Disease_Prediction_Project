import sys
from docx import Document


def docx_to_text(path):
    doc = Document(path)
    texts = [para.text for para in doc.paragraphs]
    return "\n".join(texts)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python extract_docx.py <path-to-docx>")
        sys.exit(1)
    path = sys.argv[1]
    try:
        text = docx_to_text(path)
        print(text)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
