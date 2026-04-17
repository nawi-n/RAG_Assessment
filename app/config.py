import os

from dotenv import load_dotenv

load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
HF_MODEL = "BAAI/bge-m3"
HF_TOKEN = os.getenv("HF_TOKEN")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LANGFUSE_SECRET_KEY = "sk-lf-2b77a7b3-c68a-4e7b-9942-91dc6f2021b8"
LANGFUSE_PUBLIC_KEY = "pk-lf-41443e67-2c95-46d8-8030-2fa18bfc45c5"
LANGFUSE_BASE_URL = "https://cloud.langfuse.com"
