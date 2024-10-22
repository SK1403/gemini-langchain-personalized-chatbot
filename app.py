#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Streamlit multi-persona conversational web application powered by Gemini,
#       Vertex AI, and LangChain. Integrates interactive domain persona switching,
#       RAG document uploading (PDF, TXT, MD, CSV), in-memory vector retrieval,
#       and grounded audit citations.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 04/10/2024          Saddam Khan        Initial implementation
# 22/10/2024          Saddam Khan        Enhanced persona switching UI and context document ingestion
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

"""
Explanation: Streamlit Web UI Application for the Gemini Personalized Multi-Persona Chatbot.
             Coordinates domain persona selection, RAG document uploading, in-memory FAISS
             vector indexing, streaming token responses, and grounded audit citations.
:param None: Reads execution configuration from settings and runtime user interactions
:return None: Renders interactive multi-persona web application in browser
"""

from typing import Dict, Any, List
import streamlit as st
from config import settings
from personas import PERSONAS, get_persona, Persona
from chatbot import PersonalizedGeminiChatbot
from utils.auth import resolve_gcp_project, verify_adc

def init_page_config() -> None:
    """
    Explanation:
        Sets Streamlit page layout configuration including browser title, favicon icon,
        and expanded wide layout display mode for domain persona visualization.

    :param None: Reads no input parameters.
    :return None: Configures Streamlit page display settings.
    """
    st.set_page_config(
        page_title="Gemini Personalized Chatbot (Domain + RAG)",
        page_icon="🧠",
        layout="wide",
    )

def init_session_state() -> None:
    """
    Explanation:
        Initializes Streamlit session state stores for message logs, uploaded file trackers,
        and RAG execution flags to persist across reruns.

    :param None: Initializes state dictionary keys.
    :return None: Sets default session state variables.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "uploaded_file_names" not in st.session_state:
        st.session_state.uploaded_file_names = set()
    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = False

def render_sidebar() -> Dict[str, Any]:
    """
    Explanation:
        Renders the sidebar controls for domain persona picking, editable system prompts,
        RAG knowledge upload, GCP project configuration, and conversation clearing.

    :param None: Reads user interactive inputs from Streamlit sidebar widgets.
    :return config Dict[str, Any]: Dictionary containing active user selections and uploaded files
        including 'selected_persona_id', 'active_persona', 'model_choice', 'gcp_project', etc.
    """
    with st.sidebar:
        st.header("🧠 Domain Persona Selector")

        persona_options = list(PERSONAS.keys())
        persona_labels = [f"{p.icon} {p.name}" for p in PERSONAS.values()]

        selected_index = st.selectbox(
            "Choose Domain Specialization",
            options=range(len(persona_options)),
            format_func=lambda i: persona_labels[i],
            index=0,
        )
        selected_persona_id = persona_options[selected_index]
        active_persona = get_persona(selected_persona_id)

        st.caption(f"**Focus**: {active_persona.description}")

        # Editable Persona Instruction (Collapsible)
        with st.expander("🛠️ View / Edit Persona System Prompt"):
            custom_system_prompt = st.text_area(
                "System Instruction",
                value=active_persona.system_instruction,
                height=130,
            )

        st.markdown("---")
        st.header("📚 Document Grounding (RAG)")

        enable_rag = st.toggle("Enable Document Grounding (RAG)", value=st.session_state.rag_enabled)
        st.session_state.rag_enabled = enable_rag

        uploaded_files = st.file_uploader(
            "Upload Custom Knowledge Base",
            type=["pdf", "txt", "md", "csv"],
            accept_multiple_files=True,
            help="Upload PDF policies, financial reports, or code documentation for grounded Q&A.",
        )

        st.markdown("---")
        st.header("⚙️ GCP & Model Controls")

        # ADC Status
        is_adc_valid, adc_project = verify_adc()
        if is_adc_valid:
            st.success(f"ADC Active: `{adc_project or 'Configured'}`")
        else:
            st.warning("⚠️ ADC not found. Run `gcloud auth application-default login`.")

        resolved_project = resolve_gcp_project(settings.project_id)
        gcp_project = st.text_input("GCP Project ID", value=resolved_project)
        gcp_region = st.selectbox(
            "GCP Location",
            options=["us-central1", "europe-west1", "europe-west4", "asia-northeast1", "us-east4"],
            index=0,
        )

        # Model & Temperature
        model_choice = st.selectbox(
            "Gemini Model",
            options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
            index=0 if active_persona.recommended_model == "gemini-1.5-flash" else 1,
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=active_persona.temperature,
            step=0.05,
        )

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                if "bot" in st.session_state:
                    st.session_state.bot.clear_history("streamlit_session")
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("🧹 Clear RAG", use_container_width=True):
                if "bot" in st.session_state:
                    st.session_state.bot.rag.clear()
                st.session_state.uploaded_file_names.clear()
                st.rerun()

        return {
            "selected_persona_id": selected_persona_id,
            "active_persona": active_persona,
            "custom_system_prompt": custom_system_prompt,
            "enable_rag": enable_rag,
            "uploaded_files": uploaded_files,
            "gcp_project": gcp_project,
            "gcp_region": gcp_region,
            "model_choice": model_choice,
            "temperature": temperature,
        }

def get_or_create_bot(
    selected_persona_id: str,
    active_persona: Persona,
    model_choice: str,
    gcp_project: str,
    gcp_region: str,
    temperature: float,
    custom_system_prompt: str,
) -> PersonalizedGeminiChatbot:
    """
    Explanation: Manages the lifecycle of PersonalizedGeminiChatbot instance, preserving
                 indexed RAG vector store documents across persona switches.
    :param  selected_persona_id str: Unique identifier of the selected persona
    :param  active_persona Persona: Active Persona dataclass
    :param  model_choice str: Selected Gemini model identifier
    :param  gcp_project str: Target GCP project ID
    :param  gcp_region str: Vertex AI geographical region
    :param  temperature float: Sampling temperature (0.0 to 1.0)
    :param  custom_system_prompt str: Custom system instructions for the assistant
    :return bot PersonalizedGeminiChatbot: Initialized chatbot instance
    """
    bot_key = f"{selected_persona_id}_{model_choice}_{gcp_project}_{gcp_region}_{temperature}_{hash(custom_system_prompt)}"
    if "current_bot_key" not in st.session_state or st.session_state.current_bot_key != bot_key:
        existing_rag = st.session_state.bot.rag if "bot" in st.session_state else None

        st.session_state.bot = PersonalizedGeminiChatbot(
            persona=active_persona,
            model_name=model_choice,
            project_id=gcp_project,
            location=gcp_region,
            temperature=temperature,
            system_instruction_override=custom_system_prompt,
        )
        if existing_rag:
            st.session_state.bot.rag = existing_rag
        st.session_state.current_bot_key = bot_key
    return st.session_state.bot

def handle_document_upload(bot: PersonalizedGeminiChatbot, uploaded_files: list) -> None:
    """
    Explanation: Ingests, parses, and chunks user-uploaded PDF, TXT, MD, or CSV documents into the RAG vector store.
    :param  bot PersonalizedGeminiChatbot: Active chatbot instance holding RAG engine
    :param  uploaded_files list: List of UploadedFile objects from Streamlit
    :return None: Updates RAG vector store and session state trackers
    """
    if not uploaded_files:
        return

    for file in uploaded_files:
        if file.name not in st.session_state.uploaded_file_names:
            with st.spinner(f"Indexing '{file.name}' into vector store..."):
                bytes_data = file.getvalue()
                if file.name.lower().endswith(".pdf"):
                    docs = bot.rag.load_pdf_stream(bytes_data, file.name)
                else:
                    text = bytes_data.decode("utf-8", errors="ignore")
                    docs = bot.rag.load_text_stream(text, file.name)

                count = bot.rag.add_documents(docs, file.name)
                st.session_state.uploaded_file_names.add(file.name)
                st.sidebar.toast(f"Indexed {count} chunks from {file.name}")

def render_chat_interface(
    bot: PersonalizedGeminiChatbot,
    active_persona: Persona,
    model_choice: str,
    gcp_region: str,
    enable_rag: bool,
) -> None:
    """
    Explanation: Renders persona header, recommended prompt cards, message history,
                 grounded citations, and streams real-time token responses from Gemini.
    :param  bot PersonalizedGeminiChatbot: Initialized chatbot instance
    :param  active_persona Persona: Active persona dataclass
    :param  model_choice str: Selected Gemini model name
    :param  gcp_region str: Vertex AI geographical region
    :param  enable_rag bool: Whether RAG retrieval grounding is enabled
    :return None: Renders interactive chat UI components
    """
    rag_stats = bot.rag.get_stats()
    if rag_stats["total_documents"] > 0:
        st.sidebar.info(
            f"📊 **Knowledge Base Active**:\n"
            f"- Files: `{rag_stats['total_documents']}` ({', '.join(rag_stats['files'])})\n"
            f"- Chunks: `{rag_stats['total_chunks']}` indexed\n"
            f"- Backend: `{rag_stats['embedding_backend']}`"
        )

    # Main Chat View Header
    st.title(f"{active_persona.icon} {active_persona.name}")
    status_badges = f"`Model: {model_choice}` | `Region: {gcp_region}`"
    if enable_rag and rag_stats["total_chunks"] > 0:
        status_badges += f" | `RAG Grounding: Active ({rag_stats['total_chunks']} chunks)`"
    else:
        status_badges += " | `Mode: Direct LLM Persona`"
    st.caption(status_badges)

    # Active Persona Disclaimer Banner
    if active_persona.disclaimer:
        st.caption(f"*{active_persona.disclaimer}*")

    # Suggested Sample Prompts
    if not st.session_state.messages and active_persona.sample_prompts:
        st.markdown("##### 💡 Suggested Questions:")
        cols = st.columns(len(active_persona.sample_prompts))
        for idx, prompt_text in enumerate(active_persona.sample_prompts):
            if cols[idx].button(prompt_text, key=f"sample_{idx}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt_text})
                st.rerun()

    # Render Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "citations" in msg and msg["citations"]:
                with st.expander(f"📚 Grounded Citations ({len(msg['citations'])} sources)"):
                    for cit in msg["citations"]:
                        st.markdown(f"- **Source**: `{cit.source}` (Page {cit.page})")
                        st.caption(f"> \"{cit.snippet}\"")

    # User Input Box
    user_prompt = st.chat_input("Ask a question in this domain, or ask about uploaded documents...")

    # Handle Prompt Submission (from chat_input or sample buttons)
    pending_prompt = None
    if user_prompt:
        pending_prompt = user_prompt
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
    elif st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        # Triggered via sample button click
        pending_prompt = st.session_state.messages[-1]["content"]

    if pending_prompt:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            citations_to_record = []

            try:
                for chunk in bot.stream_chat(
                    pending_prompt,
                    session_id="streamlit_session",
                    use_rag=enable_rag,
                    top_k=3,
                ):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                if full_response:
                    response_placeholder.markdown(full_response)
                    citations_to_record = list(bot.last_citations)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "citations": citations_to_record,
                    })

                    # Display Citations if available
                    if citations_to_record:
                        with st.expander(f"📚 Grounded Citations ({len(citations_to_record)} sources)"):
                            for cit in citations_to_record:
                                st.markdown(f"- **Source**: `{cit.source}` (Page {cit.page})")
                                st.caption(f"> \"{cit.snippet}\"")
                else:
                    response_placeholder.warning("Received empty response from Gemini.")

            except Exception as e:
                err_text = str(e)
                if "BILLING_DISABLED" in err_text or "requires billing" in err_text:
                    response_placeholder.error(
                        f"💳 **Billing Required**\n\n"
                        f"Vertex AI requires a Billing Account linked to project **`{bot.project_id}`**.\n\n"
                        f"👉 [Click here to enable billing on {bot.project_id}](https://console.developers.google.com/billing/enable?project={bot.project_id})"
                    )
                elif "RESOURCE_USAGE_RESTRICTION_VIOLATED" in err_text:
                    response_placeholder.error(
                        f"🔒 **Policy Restriction**\n\n"
                        f"Project **`{bot.project_id}`** has an organization policy restricting Vertex AI.\n\n"
                        f"Please switch to a sandbox/personal project in the sidebar."
                    )
                else:
                    response_placeholder.error(f"⚠️ **Inference Error**: {err_text}")

def main() -> None:
    """
    Explanation:
        Main application orchestration entry point coordinating page initialization,
        sidebar controls, bot lifecycle management, document ingestion, and chat rendering.

    :param None: Reads execution configuration and coordinates Streamlit execution cycle.
    :return None: Executes Streamlit rendering lifecycle.
    """
    init_page_config()
    init_session_state()
    cfg = render_sidebar()
    bot = get_or_create_bot(
        selected_persona_id=cfg["selected_persona_id"],
        active_persona=cfg["active_persona"],
        model_choice=cfg["model_choice"],
        gcp_project=cfg["gcp_project"],
        gcp_region=cfg["gcp_region"],
        temperature=cfg["temperature"],
        custom_system_prompt=cfg["custom_system_prompt"],
    )
    handle_document_upload(bot=bot, uploaded_files=cfg["uploaded_files"])
    render_chat_interface(
        bot=bot,
        active_persona=cfg["active_persona"],
        model_choice=cfg["model_choice"],
        gcp_region=cfg["gcp_region"],
        enable_rag=cfg["enable_rag"],
    )

if __name__ == "__main__":
    main()
