#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Retrieval-Augmented Generation (RAG) Knowledge Engine module.
#       Provides multi-format file parsing (PDF, TXT, MD, CSV), recursive semantic
#       text chunking, dual embedding backends (Vertex AI & local fallback),
#       and vector similarity search with source audit citations.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 04/10/2024          Saddam Khan        Initial implementation
# 22/10/2024          Saddam Khan        Added local document chunking and semantic context retrieval
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import io
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.embeddings import Embeddings
from langchain_google_vertexai import VertexAIEmbeddings

from config import settings
from utils.auth import resolve_gcp_project

@dataclass
class Citation:
    """
    Explanation: Data structure capturing source attribution for grounded LLM answers
    :param  source str: Original source file name or document identifier
    :param  page Optional[int]: 1-based page index where content originates
    :param  snippet str: Excerpt text snippet representing matched evidence
    :param  score Optional[float]: Similarity score or relevance ranking metric
    """
    source: str
    page: Optional[int]
    snippet: str
    score: Optional[float] = None

class SimpleFallbackEmbeddings(Embeddings):
    """
    Explanation: Lightweight deterministic word/char n-gram embedding generator.
                 Guarantees zero-dependency local RAG retrieval when cloud APIs are unconfigured.
    """
    def __init__(self, dimension: int = 128):
        """
        Explanation: Initializes local embedding projection dimension
        :param  dimension int: Fixed vector length of generated embeddings
        :return None: Instantiates embedding generator
        """
        self.dimension = dimension

    def _hash_token(self, token: str) -> int:
        """
        Explanation: Hashes individual string token into bounded integer slot
        :param  token str: Normalized word or sub-word string
        :return slot int: Modulo hash index between 0 and dimension - 1
        """
        val = 0
        for char in token:
            val = (val * 31 + ord(char)) % self.dimension
        return val

    def _embed_text(self, text: str) -> List[float]:
        """
        Explanation: Tokenizes input text, calculates frequency bag-of-words, and produces L2-normalized vector
        :param  text str: Natural language input string
        :return vector List[float]: Normalized floating point vector representation
        """
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return [0.0] * self.dimension

        counts = Counter(tokens)
        vec = [0.0] * self.dimension
        for token, count in counts.items():
            idx = self._hash_token(token)
            vec[idx] += float(count)

        # Normalize L2
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Explanation: Embeds a list of document strings in batch
        :param  texts List[str]: Collection of text strings to vectorize
        :return embeddings List[List[float]]: List of normalized embedding vectors
        """
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        """
        Explanation: Computes embedding vector for a single search query string
        :param  text str: Search query text
        :return embedding List[float]: Query vector representation
        """
        return self._embed_text(text)


def recursive_split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    separators: Optional[List[str]] = None,
) -> List[str]:
    """
    Explanation: Recursively splits document text into semantic chunks respecting natural text boundaries
    :param  text str: Raw document text to be partitioned
    :param  chunk_size int: Maximum character length per output chunk
    :param  chunk_overlap int: Number of characters to overlap between sequential chunks
    :param  separators Optional[List[str]]: Hierarchy of splitting delimiters
    :return final_chunks List[str]: List of extracted text chunks
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    final_chunks: List[str] = []
    
    def _split(text_to_split: str, sep_idx: int):
        if len(text_to_split) <= chunk_size or sep_idx >= len(separators):
            cleaned = text_to_split.strip()
            if cleaned:
                final_chunks.append(cleaned)
            return

        separator = separators[sep_idx]
        if separator == "":
            # Character slice fallback
            for i in range(0, len(text_to_split), chunk_size - chunk_overlap):
                slice_chunk = text_to_split[i : i + chunk_size].strip()
                if slice_chunk:
                    final_chunks.append(slice_chunk)
            return

        splits = text_to_split.split(separator)
        current_chunk = ""
        for part in splits:
            if not part:
                continue
            piece = (current_chunk + separator + part) if current_chunk else part
            if len(piece) <= chunk_size:
                current_chunk = piece
            else:
                if current_chunk:
                    _split(current_chunk, sep_idx + 1)
                current_chunk = part

        if current_chunk:
            _split(current_chunk, sep_idx + 1)

    _split(text, 0)
    return final_chunks


class RAGEngine:
    """
    Explanation: In-memory Retrieval-Augmented Generation engine supporting PDF, TXT, MD, CSV files.
                 Performs chunking, vector indexing, similarity retrieval, and citation tracking.
    """

    def __init__(self, project_id: str = "", location: str = "us-central1"):
        """
        Explanation: Initializes RAGEngine with GCP credentials and embedding backend
        :param  project_id str: GCP project identifier
        :param  location str: Target GCP region
        :return None: Instantiates RAG knowledge base
        """
        self.project_id = resolve_gcp_project(project_id or settings.project_id)
        self.location = location or settings.location
        self.documents: List[Document] = []
        self.indexed_files: Dict[str, int] = {}  # filename -> chunk count
        self.embedding_backend: str = "Uninitialized"
        self.vector_store: Optional[InMemoryVectorStore] = None

        self._init_embeddings()

    def _init_embeddings(self):
        """
        Explanation: Connects to Vertex AI Embeddings API, falling back to local embeddings on error
        :return None: Configures active vector store and embedding backend
        """
        if self.project_id:
            try:
                self.embeddings = VertexAIEmbeddings(
                    model_name=settings.embedding_model,
                    project=self.project_id,
                    location=self.location,
                )
                self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
                self.embedding_backend = f"Vertex AI ({settings.embedding_model})"
                return
            except Exception as e:
                pass

        # Fallback local embeddings
        self.embeddings = SimpleFallbackEmbeddings()
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
        self.embedding_backend = "Local In-Memory Embeddings (Zero Cloud Dependency)"

    def load_pdf_stream(self, file_bytes: bytes, filename: str) -> List[Document]:
        """
        Explanation: Extracts text from binary PDF stream page-by-page and segments into LangChain Documents
        :param  file_bytes bytes: Raw byte content of uploaded PDF file
        :param  filename str: Name of source document for attribution metadata
        :return docs List[Document]: List of LangChain Document objects with page metadata
        """
        docs: List[Document] = []
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        
        for page_idx, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                chunks = recursive_split_text(
                    text,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
                for chunk_idx, chunk in enumerate(chunks):
                    docs.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": filename,
                                "page": page_idx,
                                "chunk_id": f"{page_idx}_{chunk_idx}",
                            },
                        )
                    )
        return docs

    def load_text_stream(self, text_content: str, filename: str) -> List[Document]:
        """
        Explanation: Parses and chunks plain text, markdown, or CSV string into Documents
        :param  text_content str: Text content to chunk
        :param  filename str: Source filename for attribution
        :return docs List[Document]: Chunked LangChain Document collection
        """
        docs: List[Document] = []
        chunks = recursive_split_text(
            text_content,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        for chunk_idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "page": 1,
                        "chunk_id": f"1_{chunk_idx}",
                    },
                )
            )
        return docs

    def add_documents(self, new_docs: List[Document], filename: str) -> int:
        """
        Explanation: Appends parsed document chunks to knowledge repository and updates vector index
        :param  new_docs List[Document]: New Document objects to index
        :param  filename str: Source filename
        :return count int: Number of successfully indexed chunks
        """
        if not new_docs:
            return 0

        self.documents.extend(new_docs)
        self.indexed_files[filename] = len(new_docs)

        # Attempt indexing with configured embeddings, falling back on error
        try:
            self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
            self.vector_store.add_documents(self.documents)
        except Exception as e:
            # Fallback to local deterministic in-memory embeddings
            self.embeddings = SimpleFallbackEmbeddings()
            self.embedding_backend = "Local In-Memory Embeddings (Cloud Billing Pending)"
            self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
            self.vector_store.add_documents(self.documents)

        return len(new_docs)

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[str, List[Citation]]:
        """
        Explanation: Queries vector store for semantically similar chunks and formats context and citations
        :param  query str: User natural language search query
        :param  top_k int: Maximum number of closest matches to return
        :return formatted_context str: Synthesized string block for LLM prompt injection
        :return citations List[Citation]: Structured citations containing source file, page, and snippet
        """
        if not self.documents or not self.vector_store:
            return "", []

        try:
            results = self.vector_store.similarity_search(query, k=top_k)
        except Exception:
            # Fallback direct lexical search if embedding fails
            results = self._lexical_search(query, top_k=top_k)

        if not results:
            return "", []

        context_parts: List[str] = []
        citations: List[Citation] = []

        for doc in results:
            source = doc.metadata.get("source", "Document")
            page = doc.metadata.get("page", 1)
            content = doc.page_content.strip()

            context_parts.append(f"--- Document: {source} (Page {page}) ---\n{content}")
            citations.append(
                Citation(
                    source=source,
                    page=page,
                    snippet=content[:200] + ("..." if len(content) > 200 else ""),
                )
            )

        formatted_context = "\n\n".join(context_parts)
        return formatted_context, citations

    def _lexical_search(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Explanation: Fallback keyword overlap matching algorithm used when vector search is unavailable
        :param  query str: Search query string
        :param  top_k int: Number of top scoring documents to return
        :return results List[Document]: Top ranked documents based on word intersection
        """
        query_words = set(re.findall(r"\w+", query.lower()))
        scored: List[Tuple[int, Document]] = []
        for doc in self.documents:
            doc_words = set(re.findall(r"\w+", doc.page_content.lower()))
            overlap = len(query_words.intersection(doc_words))
            scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    def clear(self):
        """
        Explanation: Flushes all indexed documents, resets counters, and reinitializes vector index
        :return None: Clears knowledge base state in place
        """
        self.documents.clear()
        self.indexed_files.clear()
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)

    def get_stats(self) -> Dict[str, Any]:
        """
        Explanation: Gathers telemetry and statistics on loaded documents, chunks, and active backend
        :return stats Dict[str, Any]: Dictionary containing total_documents, total_chunks, files, and backend
        """
        return {
            "total_documents": len(self.indexed_files),
            "total_chunks": len(self.documents),
            "files": list(self.indexed_files.keys()),
            "embedding_backend": self.embedding_backend,
        }
