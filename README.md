# 🎓 YouTube Educational Assistant

A Streamlit-powered web app for transcribing, summarizing, and querying YouTube educational videos using state-of-the-art AI, embedding, and retrieval technologies.

---

## 🚀 Features

- **Transcribe YouTube Videos**: Automatically convert video audio to text using Whisper Large V3 (Groq API).
- **Semantic Summarization**: Generate concise and informative summaries of long-form video content.
- **Interactive Question & Answer**: Ask natural language questions about the transcript and get context-aware answers.
- **Efficient Retrieval**: Leverage FAISS vector search for rapid retrieval of relevant transcript segments using local embeddings.

---

## 🔧 Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Transcription**: Whisper Large V3 via [Groq API](https://groq.com/)
- **Embedding Model**: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) (local)
- **Semantic Search**: [FAISS](https://github.com/facebookresearch/faiss)
- **LLM QA/Summarization**: [LLaMA-3.1-8B-Instant](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instant)
- **Experiment Tracking/MLOps**: [MLflow](https://mlflow.org/)
- **Other**: NumPy, Requests

---

## 📦 Installation

1. **Clone the repository**
    ```
    git clone https://github.com/your-username/youtube-educational-assistant.git
    cd youtube-educational-assistant
    ```

2. **Create and activate a Python environment**
    ```
    python3 -m venv myenv
    source myenv/bin/activate  # or `.\myenv\Scripts\activate` on Windows
    ```

3. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

4. **Download/prepare models**
    - Download the BAAI embedding model locally using [sentence-transformers](https://www.sbert.net/).
    - Obtain Groq and LLM API keys and set them as environment variables:
        ```
        GROQ_API_KEY=your_api_key
        LLAMA_API_KEY=your_api_key
        ```

---

## 🏃 Usage

streamlit run app.py

text

- Paste any YouTube video URL.
- Select your desired task: **Transcription**, **Summary**, or **Q/A**.
- View, copy, and interact with transcripts or results.

---

## 📝 Project Workflow

1. **Audio Extraction**: Downloads and preprocesses YouTube audio.
2. **Transcription**: Uses Whisper Large V3 (Groq API) for robust ASR.
3. **Chunking & Embedding**: Splits transcript, converts chunks using BAAI/bge-small-en-v1.5.
4. **Vector Indexing**: Builds a FAISS index for fast semantic search.
5. **Summary & Q/A**: Summarizes and answers via LLaMA-3.1-8B-Instant; streams results in UI.
