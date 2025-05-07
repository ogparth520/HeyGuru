import streamlit as st
import json
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from ollama import Client
import requests
import tempfile

# Streamlit settings
st.set_page_config(page_title="Vachanamrut GPT", layout="centered")
st.title("🕉️ Vachanamrut GPT")
st.markdown("Ask spiritually grounded questions based on the Vachanamrut 📖")

# Load embedding model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Upload or download JSON
uploaded_file = st.file_uploader("📁 Upload your JSON file with embeddings:", type=["json"])
use_external = st.checkbox("Or load from external link")

chunks = None
if uploaded_file:
    chunks = json.load(uploaded_file)
elif use_external:
    url = st.text_input("Paste public URL to your JSON file:")
    if url:
        with st.spinner("Downloading..."):
            response = requests.get(url)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    tmp.write(response.content)
                    with open(tmp.name, "r", encoding="utf-8") as f:
                        chunks = json.load(f)
            else:
                st.error("Failed to download file. Check the link.")

# Proceed only if chunks are loaded
if chunks:
    embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")
    ids = [chunk["vachanamrut_id"] for chunk in chunks]
    titles = [chunk["title"] for chunk in chunks]
    indices = [chunk["chunk_index"] for chunk in chunks]

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    def ask_llm(context, question, sources):
        client = Client()

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
        response = client.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response['message']['content']

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

        with st.spinner("🧠 Thinking deeply..."):
            answer = ask_llm(full_context, query, sorted(used_sources))

        if any(term in answer.lower() for term in ["sahasra", "[", "]", "tetrad"]):
            st.error("⚠️ The response included an invalid citation. Please try a different question.")
        else:
            st.markdown("### 🧘🏽‍♂️ Answer")
            st.markdown(f"<div style='background-color:#f9f9f9;padding:15px;border-radius:10px'>{answer}</div>", unsafe_allow_html=True)