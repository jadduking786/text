"""
PDF RAG Chatbot
================
Upload PDF(s) -> extract text -> chunk -> embed (open-source, local, no API
key needed) -> store in a FAISS vector index -> retrieve relevant chunks for
a user question -> ask an open-weight model on Groq to answer using only the
retrieved context.

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    1. Push this repo (app.py + requirements.txt) to GitHub.
    2. On share.streamlit.io, create a new app pointing at app.py.
    3. In the app's "Secrets" settings add:
           GROQ_API_KEY = "your-groq-api-key"
       (Users can also just paste a key into the sidebar at runtime.)
"""

import os
import io
import tempfile
import traceback

import numpy as np
import streamlit as st
import faiss
from pypdf import PdfReader
from groq import Groq

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF RAG Chatbot (Groq + FAISS)",
    page_icon="📄",
    layout="wide",
)

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # open-source, runs locally via fastembed
EMBEDDING_DIM = 384

# Open-weight production models currently served on Groq.
# You can also type any other valid Groq model id in the sidebar.
GROQ_MODEL_OPTIONS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
]

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4


# --------------------------------------------------------------------------
# Cached resources
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load the local, open-source embedding model once per session."""
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=EMBEDDING_MODEL_NAME)


def get_groq_client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


# --------------------------------------------------------------------------
# PDF extraction
# --------------------------------------------------------------------------
def extract_pdf_pages(file_bytes: bytes, filename: str):
    """Return a list of {'text': str, 'source': str, 'page': int} dicts,
    one per non-empty page."""
    pages = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = text.strip()
        if text:
            pages.append({"text": text, "source": filename, "page": i + 1})
    return pages


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int, chunk_overlap: int):
    """Simple, robust sliding-window chunker that prefers to break on
    sentence/paragraph boundaries when possible."""
    text = text.strip()
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    chunks = []
    start = 0
    text_len = len(text)
    max_iterations = text_len // max(1, (chunk_size - chunk_overlap)) + 10
    iterations = 0

    while start < text_len and iterations < max_iterations:
        iterations += 1
        end = min(start + chunk_size, text_len)
        window = text[start:end]

        if end < text_len:
            # Try to end the chunk on a sentence or paragraph boundary
            best_break = -1
            for sep in ["\n\n", ". ", "\n", " "]:
                idx = window.rfind(sep)
                if idx > chunk_size * 0.4:
                    best_break = idx + len(sep)
                    break
            if best_break > 0:
                end = start + best_break
                window = text[start:end]

        window = window.strip()
        if window:
            chunks.append(window)

        if end >= text_len:
            break
        next_start = end - chunk_overlap
        start = next_start if next_start > start else end

    return chunks


def build_chunks_from_pages(pages, chunk_size, chunk_overlap):
    """Turn extracted pages into chunk records with metadata."""
    records = []
    for page in pages:
        for piece in chunk_text(page["text"], chunk_size, chunk_overlap):
            records.append(
                {
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"],
                }
            )
    return records


# --------------------------------------------------------------------------
# Embedding + FAISS index
# --------------------------------------------------------------------------
def embed_texts(model, texts):
    """Return an (n, EMBEDDING_DIM) float32 numpy array of embeddings."""
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype="float32")
    vectors = list(model.embed(texts))
    matrix = np.array(vectors, dtype="float32")
    return matrix


def build_faiss_index(embedding_matrix: np.ndarray):
    """Cosine-similarity search via normalized vectors + inner product index."""
    index = faiss.IndexFlatIP(embedding_matrix.shape[1])
    normalized = embedding_matrix.copy()
    faiss.normalize_L2(normalized)
    index.add(normalized)
    return index


def retrieve_top_k(query: str, model, index, records, k: int):
    if index is None or index.ntotal == 0:
        return []
    query_vec = embed_texts(model, [query])
    faiss.normalize_L2(query_vec)
    k = min(k, index.ntotal)
    scores, ids = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        record = dict(records[idx])
        record["score"] = float(score)
        results.append(record)
    return results


# --------------------------------------------------------------------------
# Prompting Groq
# --------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context excerpts provided below, which were retrieved from documents "
    "the user uploaded. "
    "If the answer is not contained in the context, say clearly that the "
    "documents do not contain that information — do not make anything up. "
    "When useful, mention which source/page an excerpt came from. "
    "Keep answers concise and well organized."
)


def build_user_prompt(question: str, context_chunks):
    context_blocks = []
    for i, c in enumerate(context_chunks, start=1):
        context_blocks.append(
            f"[Excerpt {i} | source: {c['source']} | page: {c['page']}]\n{c['text']}"
        )
    context_str = "\n\n".join(context_blocks) if context_blocks else "(no relevant context found)"
    return (
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above."
    )


def stream_groq_answer(client: Groq, model: str, question: str, context_chunks, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # keep a short window of prior turns for conversational context
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": build_user_prompt(question, context_chunks)})

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def init_state():
    defaults = {
        "records": [],          # chunk metadata list
        "faiss_index": None,    # faiss index object
        "processed_files": [],  # names of files already indexed
        "messages": [],         # chat history [{role, content}]
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    default_key = os.environ.get("GROQ_API_KEY", "")
    try:
        default_key = st.secrets.get("GROQ_API_KEY", default_key)
    except Exception:
        pass

    groq_api_key = st.text_input(
        "Groq API key",
        value=default_key,
        type="password",
        help="Get a free key at https://console.groq.com/keys",
    )

    model_choice = st.selectbox("Groq model", GROQ_MODEL_OPTIONS, index=0)
    custom_model = st.text_input(
        "...or type a custom Groq model id (optional)",
        value="",
        help="Overrides the dropdown above if not empty.",
    )
    selected_model = custom_model.strip() if custom_model.strip() else model_choice

    with st.expander("Advanced: chunking & retrieval"):
        chunk_size = st.slider("Chunk size (characters)", 300, 3000, DEFAULT_CHUNK_SIZE, 100)
        chunk_overlap = st.slider("Chunk overlap (characters)", 0, 600, DEFAULT_CHUNK_OVERLAP, 50)
        top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, DEFAULT_TOP_K, 1)

    st.divider()
    st.subheader("📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
    )

    process_col, clear_col = st.columns(2)
    process_clicked = process_col.button("Process", use_container_width=True)
    clear_clicked = clear_col.button("Clear KB", use_container_width=True)

    if clear_clicked:
        st.session_state.records = []
        st.session_state.faiss_index = None
        st.session_state.processed_files = []
        st.success("Knowledge base cleared.")

    if process_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            try:
                with st.spinner("Extracting text from PDF(s)..."):
                    all_pages = []
                    for f in uploaded_files:
                        file_bytes = f.getvalue()
                        pages = extract_pdf_pages(file_bytes, f.name)
                        if not pages:
                            st.warning(
                                f"No extractable text found in '{f.name}'. "
                                "It may be a scanned/image-only PDF."
                            )
                        all_pages.extend(pages)

                if not all_pages:
                    st.error("No text could be extracted from the uploaded file(s).")
                else:
                    with st.spinner("Chunking text..."):
                        records = build_chunks_from_pages(all_pages, chunk_size, chunk_overlap)

                    if not records:
                        st.error("Text was extracted but no chunks could be created.")
                    else:
                        with st.spinner("Loading embedding model (first run may take a bit)..."):
                            embed_model = load_embedding_model()

                        with st.spinner(f"Embedding {len(records)} chunks..."):
                            texts = [r["text"] for r in records]
                            matrix = embed_texts(embed_model, texts)

                        with st.spinner("Building FAISS index..."):
                            index = build_faiss_index(matrix)

                        st.session_state.records = records
                        st.session_state.faiss_index = index
                        st.session_state.processed_files = [f.name for f in uploaded_files]

                        st.success(
                            f"Indexed {len(records)} chunks from "
                            f"{len(uploaded_files)} file(s)."
                        )
            except Exception as e:
                st.error(f"Something went wrong while processing the PDF(s): {e}")
                st.code(traceback.format_exc())

    if st.session_state.processed_files:
        st.caption("Indexed files:")
        for name in st.session_state.processed_files:
            st.caption(f"• {name}")

    if st.session_state.messages:
        if st.button("🗑️ Reset conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# --------------------------------------------------------------------------
# Main chat area
# --------------------------------------------------------------------------
st.title("📄 PDF RAG Chatbot")
st.caption(
    "Open-source stack: pypdf (extraction) + fastembed (local embeddings) "
    "+ FAISS (vector search) + Groq (open-weight LLM inference)."
)

has_kb = st.session_state.faiss_index is not None and st.session_state.faiss_index.ntotal > 0

if not has_kb:
    st.info("👈 Upload PDF(s) in the sidebar and click **Process** to build the knowledge base.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(
    "Ask a question about your document(s)..." if has_kb else "Process a PDF first to start chatting",
    disabled=not has_kb,
)

if question:
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                embed_model = load_embedding_model()
                top_chunks = retrieve_top_k(
                    question, embed_model, st.session_state.faiss_index,
                    st.session_state.records, top_k,
                )

                client = get_groq_client(groq_api_key)
                answer_placeholder_generator = stream_groq_answer(
                    client, selected_model, question, top_chunks,
                    st.session_state.messages[:-1],
                )
                full_answer = st.write_stream(answer_placeholder_generator)

                if top_chunks:
                    with st.expander("📚 Sources used"):
                        for c in top_chunks:
                            st.markdown(
                                f"**{c['source']} — page {c['page']}** "
                                f"(relevance: {c['score']:.2f})"
                            )
                            st.text(c["text"][:500] + ("..." if len(c["text"]) > 500 else ""))

                st.session_state.messages.append({"role": "assistant", "content": full_answer})

            except Exception as e:
                error_msg = f"Error while generating the answer: {e}"
                st.error(error_msg)
                st.code(traceback.format_exc())
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
