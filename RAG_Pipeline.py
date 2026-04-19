import os
import pandas as pd
import numpy as np
import torch
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyMuPDFLoader
# Configure Gemini
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("API Key is not found! Make sure you have a .env file.")
genai.configure(api_key=GOOGLE_API_KEY)

print("Gemini API is configured successfully!")
# URL of the book we will use
PDF_URL = "https://www.goabroadedu.in/wp-content/uploads/2017/07/Oxford-Handbook-of-Clinical-Haematology-4e.pdf"


# Gemini response helper 
def get_response(prompt: str, model_name: str = "gemini-2.5-flash") -> str:
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    return response.text.strip()


# Pipeline class
    """
    A Retrieval-Augmented Generation (RAG) pipeline that:
      1. Loads the Oxford Handbook of Clinical Haematology PDF
      2. Splits it into chunks and embeds them with a HuggingFace model
      3. Stores / loads the FAISS vector index locally
      4. Retrieves the most relevant chunks for every user question
      5. Calls the Gemini API to generate a grounded answer
      6. Keeps a rolling conversation history (last k turns)
    """
class Pipeline:


    def __init__(
        self,
        pdf_path: str = PDF_URL,
        verbose: bool = True,
        gemini_model: str = "gemini-2.5-flash",
        embedding_model: str = "intfloat/multilingual-e5-large-instruct",
        faiss_dir: str = "./faiss_index",
        history_turns: int = 5,
    ):

        print(" Loading PDF ")
        self.documents = PyMuPDFLoader(pdf_path).load_and_split()
        print(f"Loaded {len(self.documents)} pages.")

        self.history: list[tuple[str, str]] = []
        self.k = history_turns
        self.verbose = verbose
        self.gemini_model = gemini_model
        self.faiss_dir = faiss_dir

        # Lazy-loaded 
        self.embeddings: HuggingFaceEmbeddings | None = None
        self._faiss_index: FAISS | None = None

    # ── helpers ──────────────────────────────
    def _log(self, title: str, content: str) -> None:
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"  {title}")
            print("=" * 60)
            print(content)

    # ── text splitting ────────────────────────
    def splitter(self, chunk_size: int = 1024, chunk_overlap: int = 64):
        """Split loaded documents into overlapping text chunks."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return splitter.split_documents(self.documents)

    # ── vector store (FAISS, stored locally) ──
    def vector_store(self) -> FAISS:
        """
        Build or load the FAISS vector store.
        • If a saved index exists at faiss_dir load it.
        • Otherwise embed all chunks and save to disk.
        """
        if self._faiss_index is not None:
            return self._faiss_index

        # Load embeddings model once
        if self.embeddings is None:
            print(" Loading embedding model …")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large-instruct",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
            print(" Embedding model ready.")

        if os.path.exists(self.faiss_dir):
            print(f" Loading existing FAISS index from '{self.faiss_dir}' …")
            self._faiss_index = FAISS.load_local(
                self.faiss_dir,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print(" FAISS index loaded.")
        else:
            print(" Building FAISS index from scratch …")
            texts = self.splitter()
            print(f"    → {len(texts)} chunks created, embedding now …")
            self._faiss_index = FAISS.from_documents(texts, self.embeddings)
            self._faiss_index.save_local(self.faiss_dir)
            print(f" FAISS index saved to '{self.faiss_dir}'.")

        return self._faiss_index

    # ── retrieval ─────────────────────────────
    def retrieval_with_score(
        self,
        question: str,
        k: int = 4,
        score_threshold: float = 1.0,
    ) -> str:
        """
        Retrieve the top-k most relevant document chunks for a question.

        Args:
            question:        The user's question.
            k:               Number of chunks to retrieve.
            score_threshold: Maximum L2 distance to accept
                             (lower = stricter; FAISS uses L2 by default,
                             so 1.0 is a reasonable broad threshold).

        Returns:
            A single string of concatenated chunk text used as context.
        """
        faiss_index = self.vector_store()

        # e5 models require an instruction prefix for queries
        prefixed_question = (
            "Instruct: Given a question, retrieve relevant passages\nQuery: "
            + question
        )
        matched_docs = faiss_index.similarity_search_with_score(
            prefixed_question,
            k=k,
            score_threshold=score_threshold,
        )

        context = ""
        log_chunks = ""
        for i, (doc, chunk_score) in enumerate(matched_docs):
            context += doc.page_content + "\n"
            preview = (
                doc.page_content[:300] + "…"
                if len(doc.page_content) > 300
                else doc.page_content
            )
            log_chunks += (
                f"\n[Chunk {i+1}] "
                f"Page: {doc.metadata.get('page', 'N/A')} | "
                f"L2 Score: {chunk_score:.4f}\n{preview}\n"
            )

        self._log("RETRIEVED CHUNKS", log_chunks or "(none found)")

        if not context:
            context = (
                "No relevant information found in the document; "
                "please answer based on your general medical knowledge."
            )
        return context

    # ── prompt builder ────────────────────────
    def build_prompt(
        self,
        context: str,
        history: str,
        question: str,
        template: str = "",
        patient_report="",
    ) -> str:
        """
        Assemble the full prompt string from context, history and question.

        A custom template can be supplied; otherwise the default clinical
        assistant template is used.
        """
        if patient_report:
            patient_section = f"\n\n### Patient Report:\n{patient_report}\n"
        else:
            patient_section = ""
        if not template:
            template = """
### System:
You are an empathetic and expert Clinical Haematology Assistant, specialized in the Oxford Handbook of Clinical Haematology (4th Edition). You act as a trusted bridge between complex clinical data and patient understanding — combining medical accuracy with human compassion.

---

### Core Behavior Rules:

1. **Source-First Analysis**:
   Use the provided context from the Oxford Handbook as your primary and most trusted source.
   - Never infer or assume information that is not explicitly stated in the context.
   - If the context is ambiguous, say so clearly rather than guessing.

2. **Knowledge Fallback**:
   If the required information is not in the provided context, explicitly state:
   *"This specific detail is not covered in the handbook excerpt I have, but based on established medical knowledge..."*
   Then provide an accurate, conservative answer.

3. **Accuracy-Preserving Simplification**:
   Translate clinical language into plain terms WITHOUT losing medical accuracy.
   - Good: "Myelosuppression means your bone marrow is producing fewer blood cells than normal, which can make you more prone to infections, fatigue, and bleeding."
   - Bad: "Your blood is weak." (oversimplified to the point of being misleading)
   - Critical warnings, contraindications, and side effects must ALWAYS be preserved — never omitted for the sake of simplicity.

4. **Bilingual Adaptation**:
   Detect the language of the user's question and respond in that same language.
   - English → Clear, professional, yet accessible English.
   - Arabic → Modern Standard Arabic (فصحى مبسطة) — professional but not overly complex. Avoid archaic terms.
   - Mixed input → Default to the dominant language used.

5. **Empathetic Framing**:
   Before diving into clinical information, briefly acknowledge the emotional weight of the topic when appropriate.
   - For a question about a new diagnosis: *"Receiving a diagnosis like this can feel overwhelming — here's what this means in simple terms..."*
   - For a treatment question: *"It's completely reasonable to want to understand what you're being given..."*
   - Keep acknowledgments short (1–2 sentences). Do not over-dramatize.

6. **Safety Guardrails** *(non-negotiable)*:
   - NEVER provide specific drug dosages.
   - NEVER issue a definitive diagnosis for the user's personal condition.
   - ALWAYS end every response with this disclaimer (translated to match the response language):
     >  *This information is for educational purposes only. Please consult your treating physician before making any medical decisions.*

7. **Structured Response Format**:
   Use a consistent skeleton to make every response scannable and digestible:
   - **Brief Empathetic Opening:** (1-2 sentences).
   - **Direct Answer:** The core information simplified.
   - **Details/Actionable points:** Use bullet points for symptoms, treatment phases, or precautions.
   - **Disclaimer:** The mandatory medical warning.

---

### Conversation history:
{history}

### Relevant context from the Oxford Handbook of Clinical Haematology:
{context}
###If the user's question references specific patient details, include a "Patient Report" section in the prompt with that information. Otherwise, omit this section entirely.
{report_section}
### User question:
{question}

### Clinical Assistant (Patient-Friendly Response):
"""
        return template.format(history=history, context=context, question=question)

    # ── main chat method ──────────────────────
    def llm_response(self, question: str, patient_report: str = "") -> str:
        """
        Full RAG turn:  retrieve → build prompt → call Gemini → return answer.

        Args:
            question: The user's question as a plain string.

        Returns:
            The model's answer as a plain string.
        """
        self._log("USER QUESTION", question)

        context = self.retrieval_with_score(question)

        # Build history string from stored turns
        history_str = ""
        for user_msg, ai_msg in self.history:
            history_str += f"User: {user_msg}\nAssistant: {ai_msg}\n"
        self._log("CONVERSATION HISTORY", history_str or "(no history yet)")

        prompt = self.build_prompt(
            context=context,
            history=history_str,
            question=question,
            patient_report=patient_report
        )
        self._log("FULL PROMPT SENT TO GEMINI", prompt)

        answer = get_response(prompt, model_name=self.gemini_model)
        self._log("GEMINI RESPONSE", answer)

        # Rolling history — keep only the last k turns
        self.history.append((question, answer))
        if len(self.history) > self.k:
            self.history.pop(0)

        return answer

    # ── evaluation ────────────────────────────
    def eval_pipeline(self, questions_path: str) -> float:
        """
        Evaluate the pipeline on a CSV of questions using cosine-similarity
        between the question embedding and the answer embedding.

        The CSV must have a column named 'questions'.

        Args:
            questions_path: Path to the CSV file.

        Returns:
            Percentage of questions whose answer scored above the threshold.
        """
        THRESH = 0.5
        df = pd.read_csv(questions_path)
        QA: list[tuple[np.ndarray, np.ndarray]] = []
        total_score = 0

        print(f"\n  Evaluating {len(df)} questions …")
        for q in df["questions"]:
            answer = self.llm_response(q)
            w_q = np.array(self.embeddings.embed_query(q))
            w_a = np.array(self.embeddings.embed_query(answer))
            QA.append((w_q, w_a))

        for w_q, w_a in QA:
            cos_score = cosine_similarity(
                w_q.reshape(1, -1), w_a.reshape(1, -1)
            )[0][0]
            if cos_score > THRESH:
                total_score += 1

        pct = (total_score / len(QA)) * 100
        print(f"  Evaluation complete: {pct:.1f}% of answers passed the threshold.")
        return pct


# ─────────────────────────────────────────────
# 4.  Quick demo  (run this file directly)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = Pipeline(
        pdf_path=PDF_URL,
        verbose=True,          # set False to silence debug logs
        gemini_model="gemini-1.5-flash",
    )

    print("\n" + "=" * 60)
    print("  Oxford Handbook of Clinical Haematology — RAG Chatbot")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 60 + "\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", ""}:
            print("Goodbye!")
            break
        reply = pipeline.llm_response(user_input)
        print(f"\nAssistant: {reply}\n")
