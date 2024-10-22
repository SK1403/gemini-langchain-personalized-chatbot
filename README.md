# 🧠 Gemini Personalized Enterprise Chatbot (Domain Personas + RAG)

An enterprise-grade, customizable conversational AI platform built with **Python 3.11**, **Google Cloud Vertex AI (Gemini 1.5 & 2.0)**, **LangChain**, and **Streamlit**.

This project extends standard LLM capabilities into **specialized vertical domains** (Finance, Legal, Healthcare, Cloud Architecture, Software Engineering) and adds **Retrieval-Augmented Generation (RAG)** for grounded document Q&A with source citations.

---

## 🌟 Key Features

1. **🧠 Domain Persona Specialization**:
   - **💼 Financial & Investment Analyst**: EBITDA metrics, DCF valuation concepts, risk factor disclosures, compliance disclaimers.
   - **⚖️ Corporate Legal Counsel**: IRAC structured analysis, indemnity review, contract risk detection, regulatory compliance (GDPR, CCPA).
   - **🏥 Clinical & Medical Research Assistant**: Evidence-based guideline analysis, differential workups, literature synthesis, medical disclaimer.
   - **☁️ Google Cloud Principal Architect**: Cloud Architecture Framework alignment, serverless patterns, IAM zero-trust, cost optimization.
   - **💻 Senior Software Engineer & Tech Lead**: System design, clean code, error handling, algorithmic complexity, unit test generation.
   - **🌐 General AI Assistant**: Versatile, all-purpose assistant.

2. **📚 Grounded Retrieval-Augmented Generation (RAG)**:
   - Drag-and-drop document upload (`.pdf`, `.txt`, `.md`, `.csv`).
   - Page-by-page PDF extraction and recursive character chunking (`RecursiveCharacterTextSplitter`).
   - In-memory vector database (`InMemoryVectorStore`) with **Vertex AI text-embedding-004** or dependency-free local fallback embeddings.
   - Interactive **Source Citations** accordion showing source document, page number, and snippet for complete auditability.

3. **⚡ High-Performance Streaming & Memory**:
   - Low-latency real-time token streaming with typing animation.
   - Multi-turn conversation memory (`RunnableWithMessageHistory`).
   - Hyperparameter tuning directly from sidebar (Temperature, Model, System Instruction).

4. **🚀 Cloud-Ready**:
   - Containerized with [Dockerfile](Dockerfile) ready for one-command deployment to **Google Cloud Run**.

---

## 📁 Architecture & File Layout

```
gemini-langchain-personalized-chatbot/
├── app.py              # Streamlit Web UI with Persona dropdown, RAG uploader & Citations
├── chatbot.py          # PersonalizedGeminiChatbot core engine with dual standard/RAG chains
├── personas.py         # Catalog of domain personas, guidelines, prompts, and disclaimers
├── rag_engine.py       # PDF/text loaders, recursive chunker, vector store, and citations
├── config.py           # Configuration dataclass loading from .env
├── cli.py              # Interactive terminal CLI with persona & RAG support
├── utils/
│   └── auth.py         # ADC verification and GCP project detection
├── requirements.txt    # Pinned production dependencies
├── Dockerfile          # Container specification for Cloud Run
├── .env.example        # Environment variable template
└── README.md           # Documentation
```

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Navigate to the project folder
cd gemini-langchain-personalized-chatbot

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure GCP Authentication

Ensure Google Cloud Application Default Credentials (ADC) are active:

```bash
gcloud auth application-default login
```

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Optionally specify your GCP project in `.env`:
```env
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GEMINI_MODEL=gemini-1.5-flash
```

---

## 🖥️ Running the Application

### Option A: Web Application (Streamlit)
```bash
.venv/bin/streamlit run app.py --server.port=8502
```
Open your browser to `http://localhost:8502`:
1. Select a **Domain Persona** in the sidebar.
2. (Optional) Toggle **Enable Document Grounding (RAG)** and upload a PDF or text document.
3. Click any suggested starter question or type your prompt!

### Option B: Terminal CLI
```bash
.venv/bin/python cli.py
```
1. Pick your domain from the menu `[1-6]`.
2. Enter the path to a PDF or text file (or hit Enter to skip).
3. Chat with real-time streaming output in the terminal.

---

## 🚢 Deploying to Google Cloud Run

To deploy this specialized chatbot permanently on Google Cloud:

```bash
gcloud run deploy gemini-personalized-chatbot \
  --source . \
  --region us-central1 \
  --platform managed \
  --session-affinity \
  --min-instances=0 \
  --max-instances=5 \
  --allow-unauthenticated
```
