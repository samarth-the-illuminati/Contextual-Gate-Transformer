from pathlib import Path

import pymupdf


def load_text_from_file(path: str) -> str:
    file_path = Path(path)
    extension = file_path.suffix.lower()

    if extension == ".txt":
        return file_path.read_text(encoding="utf-8")

    if extension == ".pdf":
        document = pymupdf.open(path)
        text = "\n".join(page.get_text() for page in document)
        document.close()
        return text

    raise ValueError("Only .txt and .pdf files are supported.")