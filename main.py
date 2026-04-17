from pathlib import Path

from app.logger import get_logger
from app.pipeline.ingestion_pipeline import IngestionPipeline

logger = get_logger()


def main():
    pipeline = IngestionPipeline()

    data_dir = Path("data/Mine pdfs")  # folder containing PDFs

    if not data_dir.exists():
        logger.error(f"Directory not found: {data_dir}")
        return

    pdf_files = list(data_dir.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in the directory.")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s)")

    all_results = []

    for pdf_path in pdf_files:
        logger.info(f"\nProcessing: {pdf_path.name}")

        try:
            results = pipeline.run(str(pdf_path))
            all_results.extend(results)

            logger.info(f"Processed {len(results)} chunks from {pdf_path.name}")

        except Exception as e:
            logger.error(f"Failed processing {pdf_path.name}: {e}")

    # preview few chunks
    logger.info("\n--- Sample Output ---")
    for i, (chunk, embedding) in enumerate(all_results[:3]):
        logger.info(f"Chunk {i}: {chunk[:80]}")
        logger.info(f"Embedding dim: {len(embedding)}")


if __name__ == "__main__":
    main()
