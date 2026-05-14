# 📄 PDF Bot

Talk to your PDFs like they owe you answers.

Upload any PDF.
Ask questions.
Get AI-powered answers instantly using RAG (Retrieval-Augmented Generation).

Built with:

* Streamlit
* Groq
* Sentence Transformers
* Local embeddings
* Pure sleep deprivation

---

# 🚀 Features

* 📂 Upload any PDF
* 💬 Chat with documents naturally
* 🧠 Semantic search using embeddings
* ⚡ Fast responses with Groq
* 🔍 Context-aware retrieval
* 🪄 Transparent chunk retrieval
* 🛠 Local embedding generation (no embedding API costs)

---

# 🧱 Tech Stack

| Tech                  | Purpose                 |
| --------------------- | ----------------------- |
| Streamlit             | Frontend UI             |
| Groq API              | LLM inference           |
| Sentence Transformers | Embeddings              |
| pdfplumber            | PDF text extraction     |
| NumPy                 | Similarity calculations |

---

# ⚙️ Installation

Clone the repo:

```bash
git clone https://github.com/Sathvik0077x/Pdf_Bot.git
cd Pdf_Bot
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

Run the app:

```bash
streamlit run app.py
```

---

# 🧠 How It Works

1. PDF gets uploaded
2. Text is extracted
3. Text is chunked
4. Embeddings are generated locally
5. Relevant chunks are retrieved
6. LLM answers based ONLY on document context

No random hallucinated nonsense.
Mostly.

---

# 📸 Preview

> Upload PDF → Ask Question → Get Smart Answers

Basically ChatGPT but trapped inside your PDF.

---

# 🗂 Project Structure

```bash
Pdf_Bot/
│
├── app.py
├── requirements.txt
├── .env
└── README.md
```

---

# ⚠️ Notes

* Scanned PDFs may not work properly
* First run downloads embedding model
* Keep your API key secret unless you enjoy chaos

---

# 🧃 Future Plans

* Multi-PDF chat
* Citation support
* Vector database integration
* Chat memory improvements
* Deploy online
* Probably make it look less ugly

---

# 👨‍💻 Developer

Built by Sathvik.

If it breaks, it’s probably a feature.
