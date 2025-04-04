from auto_apply_bot.model_interfaces.cover_letter_generator.cover_letter_generator import CoverLetterModelInterface
from auto_apply_bot.utils.logger import get_logger


logger = get_logger(__name__)


def main():
    job_description = (
        "We’re seeking a software engineer to join our team and help build scalable backend services using Python and AWS. "
        "Ideal candidates are passionate about cloud-native infrastructure and collaborative problem-solving."
    )

    resume_snippets = [
        "Developed a real-time data pipeline using PySpark on AWS EMR.",
        "Integrated CI/CD pipelines for backend deployments via GitHub Actions.",
        "Collaborated on scalable microservices using Flask and Docker."
    ]

    try:
        with CoverLetterModelInterface() as model:
            logger.info("Generating personalized cover letter...")

            # model.load_adapter(adapter_name)

            letter = model.generate_cover_letter(
                job_description,
                resume_snippets,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7
            )

            print("\n--- Generated Cover Letter ---\n")
            print(letter)

    except Exception as e:
        logger.error(f"Failed to run model interface: {e}")


if __name__ == "__main__":
    main()
