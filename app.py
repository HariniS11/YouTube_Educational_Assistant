import os
import time
import json
import numpy as np
import faiss
import streamlit as st
from pydub import AudioSegment
from dotenv import load_dotenv
from typing import List
from pytubefix import YouTube
from groq import Groq , APITimeoutError
import requests
from sentence_transformers import SentenceTransformer
import time
# --------------------------------------
# CONFIG
# --------------------------------------
ASR_MODEL = "whisper-large-v3"  # Groq model
# EMBED_MODEL = "thenlper/gte-small"   # very good, lightweight, and supports embeddings directly
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
ASR_SEGMENT_SECONDS = 180
TOP_K = 3
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.2

# --------------------------------------
# ENV + GROQ setup
# --------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("groq_api_key", "")
HF_API_KEY= os.getenv('HF_API_KEY','')

if not GROQ_API_KEY:
    st.warning("⚠️ Missing GROQ_API_KEY in your .env file.")
client = Groq(api_key=GROQ_API_KEY)


# --------------------------------------
# HELPERS
# --------------------------------------
def download_youtube_audio(youtube_url: str, workdir: str = 'downloads') -> str:
    os.makedirs(workdir, exist_ok=True)
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).order_by("abr").desc().first()
    if not stream:
        raise RuntimeError("No audio stream found.")
    return stream.download(output_path=workdir)

def to_wav_16k_mono(infile: str, outfile: str) -> str:
    audio = AudioSegment.from_file(infile)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(outfile, format="wav")
    return outfile

def split_wav_into_segments(wav_path: str, segment_seconds: int = ASR_SEGMENT_SECONDS) -> List[str]:
    audio = AudioSegment.from_file(wav_path)
    duration_ms = len(audio)
    seg_ms = segment_seconds * 1000
    parts = []
    idx = 0
    for start_ms in range(0, duration_ms, seg_ms):
        end_ms = min(start_ms + seg_ms, duration_ms)
        segment = audio[start_ms:end_ms]
        part_path = f"{wav_path}.part{idx}.wav"
        segment.export(part_path, format="wav")
        parts.append(part_path)
        idx += 1
    return parts

client = Groq(timeout=120.0)


# Initialize Groq client with extended timeout (e.g., 120 seconds)
client = Groq(timeout=120.0)

def transcribe_wav(filepath: str, segment_seconds: int = 60) -> str:
    st.info("🔊 Splitting audio for transcription...")
    parts = split_wav_into_segments(filepath, segment_seconds=segment_seconds)  # your existing splitting function
    full_text = ""

    for i, part in enumerate(parts):
        st.write(f"→ Transcribing segment {i+1}/{len(parts)}...")
        
        # Retry transcription with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(part, "rb") as f:
                    transcription = client.audio.transcriptions.create(
                        file=(part, f.read()),
                        model="whisper-large-v3",
                        temperature=0,
                        response_format="verbose_json",
                    )
                segment_text = transcription.text.strip()
                full_text += " " + segment_text
                break  # Success, exit retry loop

            except APITimeoutError:
                st.warning(f"Segment {i+1} transcription timed out at attempt {attempt+1}. Retrying...")
                if attempt == max_retries - 1:
                    st.error(f"Segment {i+1} failed after {max_retries} attempts due to timeout.")
            
            except Exception as e:
                st.error(f"Groq transcription failed on segment {i+1}: {e}")
                break  # For other errors, skip retry and continue to next segment

            time.sleep(2 ** attempt)  # exponential backoff wait time

    return full_text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        split_pos = text.rfind(".", start, end)
        if split_pos == -1 or split_pos < start + int(chunk_size * 0.6):
            split_pos = end
        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)
        next_start = split_pos - overlap
        if next_start <= start:
            next_start = split_pos
        start = next_start
    return chunks


@st.cache_resource(show_spinner="Loading embedding model...")
def get_local_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")

model = get_local_model()

# Cache or globally load your model for efficiency


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Converts a list of text chunks into dense embeddings using a local model.
    """
    try:
        # Batch encode all chunks (fast and consistent)
        embeddings = model.encode(texts, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
    #  st.write("Chunk embeddings shape:", embeddings.shape)     # Should be (num_chunks, 384)
    except Exception as e:
        st.warning(f"Embedding of texts failed: {e}")
        # Safe fallback: stack zero vectors (rarely needed unless catastrophic failure)
        embeddings = np.zeros((len(texts), 384), dtype=np.float32)
    return embeddings



def build_faiss(embeddings: np.ndarray):
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")

    dim = embeddings.shape[1] #number of features per vector (embedding dimension) (,384)
    index = faiss.IndexFlatIP(dim) 
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    index.add(embeddings.astype(np.float32))
    # st.write("FAISS index dimension =", index.d)
    return index  #



   

def retrieve(query: str, index: faiss.IndexFlatIP, chunk_texts: list[str], top_k: int = 5):
    try:
        # Generate local embedding for the query
        q_emb = model.encode([query], normalize_embeddings=True)  # returns shape (1, 384)
    
        # st.write("Query embedding dimension =", q_emb.shape[1])

        
        # Ensure it’s float32 (FAISS requires float32)
        q_emb = np.array(q_emb, dtype=np.float32)

    except Exception as e:
        st.warning(f"Failed to embed query locally: {e}")
        q_emb = np.zeros((1, 384), dtype=np.float32)

    # Validate FAISS dimensionality before search
    if q_emb.shape[1] != index.d:
        st.error(f"Embedding dimension mismatch! Query: {q_emb.shape[1]}, Index: {index.d}")
        st.stop()

    # Perform FAISS search
    D, I = index.search(q_emb, top_k)

    # Prepare results
    hits = [
        (chunk_texts[idx], float(D[0][i]))
        for i, idx in enumerate(I[0]) if idx != -1
    ]
    
    return hits



def generate_summary(text: str) -> str:
    prompt = f"Summarize the following text clearly:\n\n{text}"
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def generate_answer(question: str, contexts: List[str]) -> str: #https://www.llama.com/docs/how-to-guides/prompting/
    context_block = "\n\n".join(contexts)
    prompt = f"""Answer the question using only this context:
{context_block} 

Question: {question}
Answer:"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

# --------------------------------------
# STREAMLIT APP
# --------------------------------------
st.set_page_config(page_title="🎧 YouTube Educational Assistant", layout="wide")
st.title("🎓 YouTube Educational Assistant")

yt_url = st.text_input("📺 Paste YouTube URL:")
task = st.radio("Select a task:", ["Transcription", "Summary", "Q/A"])

run_task = st.button("Run Task")
if run_task:
    if not yt_url.strip():
        st.warning("Please enter a valid YouTube URL.")
        st.stop()

    try:
        st.info("🎵 Downloading audio...")
        audio_path = download_youtube_audio(yt_url, "downloads")
        wav_path = os.path.join("downloads", "audio_16k.wav")
        to_wav_16k_mono(audio_path, wav_path)
        st.success("Audio ready ✅")

        st.info("🧠 Transcribing using Groq Whisper...")
        transcript = transcribe_wav(wav_path)
        st.session_state.transcript = transcript  # Save it here
      

        if "index" in st.session_state:
            del st.session_state.index
        if "chunks" in st.session_state:
            del st.session_state.chunks

        st.success("Transcription complete ✅")
        
    except Exception as e:
        st.error(f"Error: {e}")    

# display blocks after transcription exists
if "transcript" in st.session_state:
    transcript = st.session_state.transcript

    if task == "Transcription":
        st.subheader("📜 Transcript")
        st.text_area("Transcript Text:", transcript, height=300)

    elif task == "Summary":
        st.subheader("🪄 Summary")
        summary = generate_summary(transcript)
        st.success(summary)

    elif task == "Q/A":
        st.subheader("❓ Ask Questions about Transcript")

        # Build index if not yet done
        if "index" not in st.session_state:
            chunks = chunk_text(transcript)
            st.session_state.chunks = chunks
            st.info("🔢 Creating embeddings & FAISS index...")
            embeddings = embed_texts(chunks)
            # st.write(embeddings.shape)
            st.session_state.index = build_faiss(embeddings)
            
            st.success("Embeddings stored in FAISS ✅")
        
        question = st.text_input("Ask your question:")
       
        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                hits = retrieve(question, st.session_state.index, st.session_state.chunks)

                if not hits:
                    st.error("No relevant context found. Try a different question.")
                else:
                    top_contexts = [h[0] for h in hits]
                    answer = generate_answer(question, top_contexts)
                    st.write("### 💬 Answer:")
                    st.success(answer)
