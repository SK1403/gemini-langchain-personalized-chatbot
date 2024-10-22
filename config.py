#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Configuration module encapsulating environment variables, GCP project and
#       region defaults, Gemini and embedding model selections, and RAG chunking parameters.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 04/10/2024          Saddam Khan        Initial implementation
# 22/10/2024          Saddam Khan        Added runtime model and persona configuration settings
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

@dataclass
class Settings:
    """
    Explanation: Dataclass holding application configuration attributes, model IDs, and chunking hyperparameters
    :param  project_id str: Default GCP Project ID
    :param  location str: Default GCP region for Vertex AI endpoints
    :param  model_name str: Target Gemini generative model identifier
    :param  embedding_model str: Target Vertex AI text embedding model
    :param  temperature float: Default temperature setting for LLM responses
    :param  max_output_tokens int: Upper bound of token generation per turn
    :param  chunk_size int: Target character count for RAG text chunks
    :param  chunk_overlap int: Overlap character count between adjacent chunks
    """
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    location: str = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    model_name: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.7"))
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))

settings = Settings()
