from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq


# Project root
BASE_DIR = Path(__file__).resolve().parents[2]


def get_groq_key():
    """
    Find GROQ API key from .env / api_key.env.
    """

    # Load .env
    load_dotenv(
        BASE_DIR / ".env"
    )

    # Load api_key.env
    load_dotenv(
        BASE_DIR / "api_key.env"
    )

    # First try environment variable
    key = os.getenv(
        "GROQ_API_KEY"
    )

    if key:
        return key.strip()


    # Fallback: manually read api_key.env
    env_file = BASE_DIR / "api_key.env"

    if env_file.exists():

        content = env_file.read_text(
            encoding="utf-8"
        ).strip()

        for line in content.splitlines():

            line = line.strip()

            if line.startswith(
                "GROQ_API_KEY="
            ):

                return (
                    line.split(
                        "=",
                        1
                    )[1]
                    .strip()
                    .strip('"')
                    .strip("'")
                )

    return None


def get_llm():

    api_key = get_groq_key()

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY not found. "
            "Please check api_key.env"
        )

    return ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    groq_api_key=api_key,
)


def ask_ai(prompt: str):
    """
    Send prompt to Groq and return text response.
    """

    llm = get_llm()

    response = llm.invoke(
        prompt
    )

    return response.content
