from auto_apply_bot.model_interfaces.coverletter_generator.cover_letter_generator import CoverLetterModelInterface
from auto_apply_bot import resolve_project_source
from auto_apply_bot.logger import get_logger


logger = get_logger(__name__)


# MARK This is not a good or usable function, this is not production code, as this project is just a for fun tool for me I am leaving in some scripts I have used to query the 
#  model before I do anything more complicated with it
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

    with CoverLetterModelInterface(mode="inference") as model:
        logger.info("Generating personalized cover letter...")
        letter = model.generate_cover_letter(job_description, 
                                             resume_snippets, 
                                             max_new_tokens=2048, 
                                             )
        print("\n--- Generated Cover Letter ---\n")
        print(letter)

if __name__ == "__main__":
    main()
