from auto_apply_bot.model_interfaces.coverletter_generator.cover_letter_generator import CoverLetterModelInterface
from auto_apply_bot import resolve_project_source
from pathlib import Path
from auto_apply_bot.logger import get_logger


logger = get_logger(__name__)


def main():
    rag_element_dir = resolve_project_source() / "demo_RAG_files"
    if not rag_element_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {rag_element_dir}")

    file_paths = [p for p in rag_element_dir.iterdir() if p.is_file()]
    if not file_paths:
        raise FileNotFoundError(f"No files found in: {rag_element_dir}")

    logger.info(f"Found {len(file_paths)} training files.")

    with CoverLetterModelInterface(mode="training") as model_interface:
        output_path = model_interface.train_on_existing_letters(
            [str(p) for p in file_paths],
        )
        logger.info(f"LoRA model fine-tuned and saved at: {output_path}")

if __name__ == "__main__":
    main()