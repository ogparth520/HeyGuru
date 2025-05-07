import streamlit as st
import json
import numpy as np
import faiss
import re
import requests
import tempfile
from sentence_transformers import SentenceTransformer
from openai import OpenAI

st.set_page_config(page_title="HeyGuru!", layout="centered")

# 🌟 Full visual theme styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #faf9f6;
        color: #222222;
        font-family: "Georgia", serif;
    }

    h1, h2, p, label, input, textarea {
        color: #222222 !important;
        font-family: "Georgia", serif !important;
    }

    .block-container {
        padding-top: 2rem;
    }

    hr {
        border: none;
        border-top: 1px solid #e0e0e0;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }

    .upload-section, .answer-box {
        background-color: #f7f7f7;
        border-radius: 10px;
        padding: 20px;
        font-size: 1.05rem;
        line-height: 1.7;
        color: #222222;
        font-family: Georgia, serif;
    }

    .css-1cpxqw2.edgvbvh3 {  /* Fixes button alignment if needed */
        color: #d97706;
    }

    .stButton>button {
        background-color: #d97706 !important;
        color: white !important;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Beautiful centered header
st.markdown(
    """
    <div style='text-align: center; padding-top: 10px;'>
        <h1 style='font-size: 3em;'>HeyGuru!</h1>
        <p style='font-size: 1.2em; color: #555;'>Ask spiritually grounded questions based on the Vachanamrut</p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

uploaded_file = st.file_uploader("📁 Upload your JSON file with embeddings:", type=["json"])
use_external = st.checkbox("Or load from external link")

chunks = None
if uploaded_file:
    chunks = json.load(uploaded_file)
elif use_external:
    url = st.text_input("Paste public URL to your JSON file:")
    if url:
        with st.spinner("📥 Downloading..."):
            response = requests.get(url)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(response.content)
                    with open(tmp.name, "r", encoding="utf-8") as f:
                        chunks = json.load(f)
            else:
                st.error("Failed to download file. Check the link.")

if chunks:
    embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")
    ids = [chunk["vachanamrut_id"] for chunk in chunks]
    titles = [chunk["title"] for chunk in chunks]
    indices = [chunk["chunk_index"] for chunk in chunks]

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    def ask_llm(context, question, sources):
        client = OpenAI()
        system_message = """
You are a disciplined Swaminarayan scholar.

Rules:
- Use only the context and sources provided.
- Cite at least one Vachanamrut inline (e.g., “as stated in Gadhadā I-11”).
- Do not use brackets like [1], [Gadhadā].
- Do not invent titles or quote unknown scriptures.
- Speak in a warm, pravachan tone.
"""

        user_prompt = f"""
Context:
{context}

Sources:
{chr(10).join(f"- {s}" for s in sources)}

Question:
{question}

Answer:
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    query = st.text_input("Ask a question:")
    if query:
        query_embedding = model.encode(query).astype("float32").reshape(1, -1)
        distances, indices_found = index.search(query_embedding, 5)

        all_context_chunks = []
        used_sources = set()

        for i, idx in enumerate(indices_found[0]):
            similarity_score = 1 / (1 + distances[0][i])
            if similarity_score < 0.60:
                continue
            match_id = ids[idx]
            chunk_index = indices[idx]
            source_title = f"{match_id} – {titles[idx]}"
            used_sources.add(source_title)

            for offset in [-2, -1, 0, 1, 2]:
                neighbor_idx = chunk_index + offset
                for c in chunks:
                    if c["vachanamrut_id"] == match_id and c["chunk_index"] == neighbor_idx:
                        all_context_chunks.append(c["text"])

        full_context = " ".join(all_context_chunks).replace("\n", " ").strip()
        full_context = re.sub(r"\s{2,}", " ", full_context)

        with st.spinner("🧘🏽‍♂️ Thinking deeply..."):
            answer = ask_llm(full_context, query, sorted(used_sources))

        if any(term in answer.lower() for term in ["sahasra", "[", "]", "tetrad"]):
            st.error("⚠️ The response included an invalid citation. Please try a different question.")
        else:
            st.markdown("### 💬 Answer")
            st.markdown(
                f"""
                <div class="answer-box">
                {answer}
                </div>
                """,
                unsafe_allow_html=True
            )