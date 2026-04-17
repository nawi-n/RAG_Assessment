from typing import List

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.logger import get_logger

logger = get_logger()


def is_table_block(text: str) -> bool:
    lines = text.strip().split("\n")

    if len(lines) < 2:
        return False

    pipe_lines = [line for line in lines if line.count("|") >= 2]

    return len(pipe_lines) >= 2


def chunk_text(text: str) -> List[str]:
    logger.info("Starting STRUCTURED chunking (table-aware)")

    # =========================
    # STEP 1: Split by headers
    # =========================
    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    docs = markdown_splitter.split_text(text)

    # =========================
    # STEP 2: Separate tables vs text
    # =========================
    structured_blocks = []

    for doc in docs:
        content = doc.page_content.strip()

        # 🔥 Keep tables intact
        if is_table_block(content):
            structured_blocks.append(content)
        else:
            structured_blocks.append(content)

    # =========================
    # STEP 3: Chunk ONLY non-table text
    # =========================
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 🔥 increased
        chunk_overlap=150,  # 🔥 better context retention
    )

    final_chunks = []

    for block in structured_blocks:
        if is_table_block(block):
            # 🚀 keep full table as ONE chunk
            final_chunks.append(block)
        else:
            split_chunks = recursive_splitter.split_text(block)
            final_chunks.extend(split_chunks)

    logger.info(f"Generated {len(final_chunks)} chunks (table-safe)")

    return final_chunks
