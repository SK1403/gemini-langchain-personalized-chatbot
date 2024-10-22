#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Personalized Conversational Engine combining domain personas (Finance,
#       Legal, Healthcare, Cloud Architecture, Software Engineering) with
#       document grounding via Retrieval-Augmented Generation (RAG) and LangChain.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 04/10/2024          Saddam Khan        Initial implementation
# 22/10/2024          Saddam Khan        Added dynamic system persona injection and memory engine
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from typing import Generator, Dict, Any, List, Optional
from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from config import settings
from personas import Persona, get_persona
from rag_engine import RAGEngine, Citation
from utils.auth import resolve_gcp_project

class PersonalizedGeminiChatbot:
    """
    Explanation: Multi-Persona Conversational Engine leveraging Gemini on Vertex AI and LangChain.
                 Supports domain persona selection, multi-turn memory, RAG document grounding,
                 and streaming responses.
    """

    def __init__(
        self,
        persona: Optional[Persona] = None,
        model_name: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        temperature: Optional[float] = None,
        system_instruction_override: Optional[str] = None,
    ):
        """
        Explanation: Initializes personalized chatbot with persona parameters, Vertex AI LLM, and dual chains
        :param  persona Optional[Persona]: Domain persona object defining tone, temperature, and instructions
        :param  model_name Optional[str]: Gemini model identifier overriding persona default
        :param  project_id Optional[str]: GCP project ID hosting Vertex AI
        :param  location Optional[str]: GCP region for model endpoints
        :param  temperature Optional[float]: Sampling temperature parameter
        :param  system_instruction_override Optional[str]: Custom prompt overriding persona default
        :return None: Instantiates PersonalizedGeminiChatbot object
        """
        self.persona: Persona = persona or get_persona("general")
        self.project_id = resolve_gcp_project(project_id or settings.project_id)
        self.location = location or settings.location
        self.model_name = model_name or self.persona.recommended_model or settings.model_name
        self.temperature = temperature if temperature is not None else self.persona.temperature
        self.system_instruction = (
            system_instruction_override.strip()
            if system_instruction_override and system_instruction_override.strip()
            else self.persona.system_instruction
        )

        # In-memory session store: session_id -> InMemoryChatMessageHistory
        self._session_store: Dict[str, InMemoryChatMessageHistory] = {}

        # RAG Knowledge Engine
        self.rag = RAGEngine(project_id=self.project_id, location=self.location)
        self.last_citations: List[Citation] = []

        # Initialize Vertex AI LLM
        self.llm = ChatVertexAI(
            model=self.model_name,
            project=self.project_id if self.project_id else None,
            location=self.location,
            temperature=self.temperature,
            max_output_tokens=settings.max_output_tokens,
            max_retries=1,
        )

        # Build Chains
        self._build_chains()

    def _build_chains(self):
        """
        Explanation: Constructs standard conversational chain and RAG grounded chain with system prompts
        :return None: Attaches runnable conversational chains to instance
        """
        # Standard Conversational Prompt
        self.standard_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_instruction),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        self.standard_chain = self.standard_prompt | self.llm

        # RAG Grounded Conversational Prompt
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                f"{self.system_instruction}\n\n"
                "=== DOCUMENT GROUNDING CONTEXT ===\n"
                "{context}\n"
                "==================================\n\n"
                "Grounding Rules:\n"
                "1. Prioritize answering based on the provided Document Grounding Context above.\n"
                "2. If the context contains relevant facts, synthesize them clearly and cite the document/page.\n"
                "3. If the context does not contain the answer, state that your uploaded documents do not address it, "
                "then provide an answer using your general domain expertise.\n"
                "4. Maintain your assigned persona tone and formatting guidelines."
            )),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        self.rag_chain = self.rag_prompt | self.llm

        # Wrapped Chains with Session History
        self.conversational_standard_chain = RunnableWithMessageHistory(
            self.standard_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        self.conversational_rag_chain = RunnableWithMessageHistory(
            self.rag_chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Explanation: Retrieves or initializes an in-memory session chat message history
        :param  session_id str: Unique identifier representing user session
        :return history BaseChatMessageHistory: LangChain chat message history container
        """
        if session_id not in self._session_store:
            self._session_store[session_id] = InMemoryChatMessageHistory()
        return self._session_store[session_id]

    def stream_chat(
        self,
        user_input: str,
        session_id: str = "default_session",
        use_rag: bool = False,
        top_k: int = 3,
    ) -> Generator[str, None, None]:
        """
        Explanation: Streams response tokens chunk-by-chunk using either RAG or standard persona chain
        :param  user_input str: Natural language input prompt from user
        :param  session_id str: Conversation session identifier for context memory
        :param  use_rag bool: Toggle indicating whether to perform document retrieval
        :param  top_k int: Number of top relevant document chunks to inject as context
        :return chunk Generator[str, None, None]: Stream of string text tokens
        """
        self.last_citations = []
        context = ""

        if use_rag and self.rag.documents:
            context, citations = self.rag.retrieve(user_input, top_k=top_k)
            self.last_citations = citations

        if use_rag and context:
            # Stream with RAG chain
            for chunk in self.conversational_rag_chain.stream(
                {"input": user_input, "context": context},
                config={"configurable": {"session_id": session_id}},
            ):
                if chunk.content:
                    yield chunk.content
        else:
            # Stream with standard persona chain
            for chunk in self.conversational_standard_chain.stream(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}},
            ):
                if chunk.content:
                    yield chunk.content

    def chat(
        self,
        user_input: str,
        session_id: str = "default_session",
        use_rag: bool = False,
        top_k: int = 3,
    ) -> str:
        """
        Explanation: Sends input prompt and returns complete concatenated response string
        :param  user_input str: Natural language input prompt from user
        :param  session_id str: Conversation session identifier
        :param  use_rag bool: Whether to enable RAG document retrieval
        :param  top_k int: Number of retrieved chunks to consider
        :return response str: Complete generated text response
        """
        chunks = list(self.stream_chat(user_input, session_id, use_rag, top_k))
        return "".join(chunks)

    def get_history(self, session_id: str = "default_session") -> List[BaseMessage]:
        """
        Explanation: Retrieves chronological message history for the requested session
        :param  session_id str: Conversation session identifier
        :return messages List[BaseMessage]: Stored human and AI messages
        """
        return self._get_session_history(session_id).messages

    def clear_history(self, session_id: str = "default_session") -> None:
        """
        Explanation: Purges conversation history and resets session memory store
        :param  session_id str: Conversation session identifier to clear
        :return None: Empties session history in place
        """
        if session_id in self._session_store:
            self._session_store[session_id].clear()
