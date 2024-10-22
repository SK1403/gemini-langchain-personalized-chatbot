#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Command-line interface (CLI) for Gemini Personalized Chatbot.
#       Allows interactive domain persona selection, file ingestion commands,
#       RAG context toggling, and streamed terminal responses.
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 04/10/2024          Saddam Khan        Initial implementation
# 22/10/2024          Saddam Khan        Added terminal persona selection and streaming chat loop
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import sys
import os
from personas import PERSONAS, get_persona
from chatbot import PersonalizedGeminiChatbot
from utils.auth import verify_adc

def main():
    """
    Explanation: Entrypoint for running interactive personalized CLI chat loop
    :return None: Executes CLI loop until exit command or interruption
    """
    print("=" * 70)
    print("      Gemini Personalized Chatbot (Domain Personas + RAG CLI)")
    print("=" * 70)

    # 1. ADC Check
    is_valid, project_info = verify_adc()
    if not is_valid:
        print("\n[!] WARNING: Google ADC credentials not detected.")
        print("    Run: gcloud auth application-default login")
        print("    Details:", project_info)
        print("=" * 70)

    # 2. Persona Selection
    print("\nSelect Domain Specialization:")
    persona_list = list(PERSONAS.values())
    for idx, p in enumerate(persona_list, start=1):
        print(f"  [{idx}] {p.icon} {p.name} - {p.description[:50]}...")

    choice = input(f"\nSelect persona [1-{len(persona_list)}] (default 1): ").strip()
    try:
        chosen_idx = int(choice) - 1 if choice else 0
        if not (0 <= chosen_idx < len(persona_list)):
            chosen_idx = 0
    except ValueError:
        chosen_idx = 0

    active_persona = persona_list[chosen_idx]
    print(f"\n[*] Activated Persona: {active_persona.icon} {active_persona.name}")
    if active_persona.disclaimer:
        print(f"[*] Note: {active_persona.disclaimer}")

    # 3. Initialize Bot
    try:
        bot = PersonalizedGeminiChatbot(persona=active_persona)
        print(f"[*] GCP Project: {bot.project_id or '(auto-detected via ADC)'}")
        print(f"[*] GCP Region:  {bot.location}")
        print(f"[*] Model:       {bot.model_name}")
        print(f"[*] Temperature: {bot.temperature}")
    except Exception as e:
        print(f"[x] Error initializing chatbot: {e}")
        sys.exit(1)

    # 4. Optional Document Ingestion for RAG
    doc_path = input("\nEnter path to a PDF/TXT/MD file for RAG grounding (or press Enter to skip): ").strip()
    use_rag = False
    if doc_path and os.path.exists(doc_path):
        filename = os.path.basename(doc_path)
        print(f"[*] Indexing '{filename}' into vector store...")
        with open(doc_path, "rb") as f:
            data = f.read()
        if doc_path.lower().endswith(".pdf"):
            docs = bot.rag.load_pdf_stream(data, filename)
        else:
            docs = bot.rag.load_text_stream(data.decode("utf-8", errors="ignore"), filename)
        count = bot.rag.add_documents(docs, filename)
        print(f"[✓] Successfully indexed {count} chunks using {bot.rag.embedding_backend}!")
        use_rag = True

    session_id = "cli_personalized_session"
    print("\nCommands: 'exit' to quit, 'clear' to reset history, 'rag' to toggle RAG.")
    print("-" * 70)

    # 5. Chat Loop
    while True:
        try:
            rag_indicator = "[RAG ON]" if (use_rag and bot.rag.documents) else "[RAG OFF]"
            user_input = input(f"\nYou {rag_indicator}: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            if user_input.lower() in ["clear", "reset"]:
                bot.clear_history(session_id)
                print("[*] Session history cleared.")
                continue

            if user_input.lower() in ["rag"]:
                use_rag = not use_rag
                print(f"[*] RAG is now {'ENABLED' if use_rag else 'DISABLED'}.")
                continue

            print(f"\n{active_persona.name}: ", end="", flush=True)
            for chunk in bot.stream_chat(user_input, session_id=session_id, use_rag=use_rag):
                print(chunk, end="", flush=True)
            print()

            if use_rag and bot.last_citations:
                print("\n  [Citations]")
                for cit in bot.last_citations:
                    print(f"  - Source: {cit.source} (Page {cit.page})")
                    print(f"    Snippet: \"{cit.snippet[:120]}...\"")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n[x] Error: {e}")

if __name__ == "__main__":
    main()
