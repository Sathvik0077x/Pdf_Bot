# app.py
# Chat With Any PDF — RAG powered document assistant
# Run with: streamlit run app.py

import streamlit as st
import pdfplumber
import tempfile
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# ── SETUP ──────────────────────────────────────────────
load_dotenv()

st.write("App started")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load embedding model once
# Runs on your laptop — no API needed, no cost
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embed_model()

# ── PAGE CONFIG ─────────────────────────────────────────
st.set_page_config(
    page_title="Chat With Any PDF",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Chat With Any PDF")
st.write("Upload a PDF → Ask anything → Get answers from your document")
st.divider()

# ── SESSION STATE ───────────────────────────────────────
# Streamlit reruns the entire script on every interaction
# session_state survives reruns — this is how we keep data

if "chunks" not in st.session_state:
    st.session_state.chunks = []        # PDF text chunks

if "conversation" not in st.session_state:
    st.session_state.conversation = []  # Chat history

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None    # Uploaded PDF name

# ── HELPER FUNCTIONS ────────────────────────────────────

def extract_text_from_pdf(uploaded_file):
    """
    Extract all text from an uploaded PDF file.
    Uses a temp file because pdfplumber needs a file path.
    """
    text = ""
    
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    
    # Extract text page by page
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Some pages might be empty
                text += page_text + "\n"
    
    # Delete temp file — clean up after ourselves
    os.unlink(tmp_path)
    
    return text


def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split long text into smaller overlapping chunks.
    
    Why chunk? You can't send a 50 page PDF to the AI at once.
    Why overlap? If an answer spans two chunks, overlap captures it.
    
    chunk_size=500: each chunk is ~500 characters
    overlap=50: consecutive chunks share 50 characters
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():  # Don't add empty chunks
            chunks.append(chunk)
        
        start = end - overlap  # Move forward but keep overlap
    
    return chunks


def find_relevant_chunks(question, chunks, top_k=3):
    """
    Find the top_k chunks most relevant to the question.
    Uses cosine similarity between embeddings.
    """
    # Embed everything
    question_embedding = embed_model.encode(question)
    chunk_embeddings = embed_model.encode(chunks)
    
    # Calculate similarity between question and every chunk
    similarities = []
    for chunk_emb in chunk_embeddings:
        similarity = np.dot(question_embedding, chunk_emb) / (
            np.linalg.norm(question_embedding) *
            np.linalg.norm(chunk_emb)
        )
        similarities.append(similarity)
    
    # Get indices of top_k highest scores
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [chunks[i] for i in top_indices]


def generate_answer(question, relevant_chunks, conversation_history):
    """
    Generate an answer using retrieved chunks as context.
    Includes conversation history for follow-up questions.
    """
    # Combine top chunks into one context block
    context = "\n\n---\n\n".join(relevant_chunks)
    
    # Build messages list
    messages = [
        {
            "role": "system",
            "content": """You are a precise document assistant.

STRICT RULES:
1. Answer ONLY from the provided document context
2. If the answer is not in the context, say exactly:
   "I could not find that information in this document."
3. Never make up information
4. Keep answers clear and concise
5. Reference the relevant part of the document in your answer"""
        }
    ]
    
    # Add conversation history for follow-up question support
    messages.extend(conversation_history)
    
    # Add current question with context
    messages.append({
        "role": "user",
        "content": f"""Document context:
{context}

Question: {question}

Answer based strictly on the context above:"""
    })
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=500,
        temperature=0.1  # Low = factual, consistent, grounded
    )
    
    return response.choices[0].message.content


# ── LAYOUT ──────────────────────────────────────────────

left_col, right_col = st.columns([1, 2])

# ── LEFT COLUMN: Upload ─────────────────────────────────

with left_col:
    st.subheader("1. Upload Your PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload any PDF — resume, textbook, report, policy document"
    )
    
    if uploaded_file is not None:
        if st.button("Process PDF", type="primary", use_container_width=True):
            
            with st.spinner("Reading PDF and preparing AI..."):
                
                # Step 1: Extract text
                text = extract_text_from_pdf(uploaded_file)
                
                if not text.strip():
                    st.error("Could not extract text. PDF might be a scanned image.")
                
                else:
                    # Step 2: Chunk the text
                    chunks = chunk_text(text)
                    
                    # Step 3: Store in session
                    st.session_state.chunks = chunks
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.conversation = []  # Reset for new PDF
                    
                    st.success(f"Ready! Created {len(chunks)} searchable chunks.")
    
    # Show current PDF info
    if st.session_state.pdf_name:
        st.divider()
        st.info(f"📎 **Active:** {st.session_state.pdf_name}")
        st.write(f"**Chunks:** {len(st.session_state.chunks)}")
        st.write(f"**Messages:** {len(st.session_state.conversation)}")
        
        if st.button("Clear & Upload New", use_container_width=True):
            st.session_state.chunks = []
            st.session_state.conversation = []
            st.session_state.pdf_name = None
            st.rerun()

# ── RIGHT COLUMN: Chat ──────────────────────────────────

with right_col:
    st.subheader("2. Ask Questions")
    
    if not st.session_state.chunks:
        st.info("👈 Upload and process a PDF first to start chatting")
    
    else:
        # Show conversation history
        for message in st.session_state.conversation:
            
            if message["role"] == "user":
                with st.chat_message("user"):
                    # User messages have context appended — show only the question
                    content = message["content"]
                    if "Document context:" in content:
                        question_part = content.split("Question: ")[-1].split("\n")[0]
                        st.write(question_part)
                    else:
                        st.write(content)
            
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(message["content"])
        
        # Chat input box
        question = st.chat_input("Ask anything about the document...")
        
        if question:
            
            # Show user message immediately
            with st.chat_message("user"):
                st.write(question)
            
            # Generate and show answer
            with st.chat_message("assistant"):
                with st.spinner("Searching document and thinking..."):
                    
                    try:
                        # Find relevant chunks
                        relevant_chunks = find_relevant_chunks(
                            question,
                            st.session_state.chunks
                        )
                        
                        # Generate answer
                        answer = generate_answer(
                            question,
                            relevant_chunks,
                            st.session_state.conversation
                        )
                        
                        # Display answer
                        st.write(answer)
                        
                        # Show retrieved context (transparency feature)
                        with st.expander("📎 View retrieved context"):
                            for i, chunk in enumerate(relevant_chunks):
                                st.write(f"**Chunk {i+1}:**")
                                st.write(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                                st.divider()
                    
                    except Exception as e:
                        st.error(f"Something went wrong: {e}")
                        answer = "Error generating answer."
            
            # Save to conversation history
            st.session_state.conversation.append({
                "role": "user",
                "content": f"Document context:\n{chr(10).join(relevant_chunks)}\n\nQuestion: {question}\n\nAnswer based strictly on the context above:"
            })
            st.session_state.conversation.append({
                "role": "assistant",
                "content": answer
            })