# app/parsers/pdf_parser.py
import pymupdf4llm

from app.logger import get_logger

logger = get_logger()


def parse_pdf(file_path: str) -> str:
    logger.info(f"Parsing PDF with pymupdf4llm: {file_path}")

    try:
        # Converts entire PDF → clean markdown
        markdown_text = pymupdf4llm.to_markdown(file_path)

        logger.debug("PDF successfully converted to markdown")

    except Exception as e:
        logger.error(f"Error parsing PDF: {e}")
        raise

    return markdown_text
