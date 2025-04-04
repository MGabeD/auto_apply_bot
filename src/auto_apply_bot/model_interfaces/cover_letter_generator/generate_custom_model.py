from auto_apply_bot.model_interfaces.cover_letter_generator.cover_letter_generator import CoverLetterModelInterface
from auto_apply_bot import resolve_project_source
from auto_apply_bot.utils.logger import get_logger
from pathlib import Path

logger = get_logger(__name__)

# NOTE: This is a personal script meant for testing/training purposes and is not production code.
def main():
    rag_element_dir: Path = resolve_project_source() / "demo_RAG_files"
    if not rag_element_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {rag_element_dir}")

    file_paths = [str(p) for p in rag_element_dir.iterdir() if p.is_file()]
    if not file_paths:
        raise FileNotFoundError(f"No files found in: {rag_element_dir}")

    logger.info(f"Found {len(file_paths)} training files.")

    with CoverLetterModelInterface() as model_interface:
        output_path = model_interface.train_on_existing_letters(letter_paths=file_paths)
        if output_path:
            logger.info(f"LoRA model fine-tuned and saved at: {output_path}")
        else:
            logger.warning("Training skipped or failed. No model saved.")

if __name__ == "__main__":
    main()
