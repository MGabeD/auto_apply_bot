
This is an exploratory project focused on automating the job application process using LLMs. The system aims to (1) assess how well a candidate's experience aligns with a given job posting, and (2) generate high-quality, personalized cover letters tailored to both the job description and the candidate’s past writing style.

The generation pipeline leverages a RAG (Retrieval-Augmented Generation) workflow to surface relevant experience snippets, and then uses prompt-based generation to create customized output. The model is fine-tuned using LoRA adapters on top of a base language model (e.g., DeepSeek-7B), allowing for lightweight personalization based on previous cover letters or interactive feedback sessions.

The project supports both training and inference modes, interactive RLHF-style feedback loops, and modular plugin support for different embedding models, retrieval strategies, or base LLMs.

Whether you're experimenting with personalized AI writing or building tools to streamline job hunting, this bot serves as a flexible and extensible foundation.


Instructions for usage

1) pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
2) Choose Between the following depending on your operating system

    A) python -m pip install -e .[test,unix]

    B) python -m pip install -e .[test,windows]
 