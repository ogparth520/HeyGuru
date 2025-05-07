import streamlit as st
import json
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from ollama import Client

# Load Vachanamrut chunks + embeddings
@st.cache_resource
def load_data():
    with open("vachanamrut_chunks_with_embeddings.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")
    return chunks, embeddings

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

chunks, embeddings = load_data()
model = load_model()
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# LLM wrapper with system-level constraints
def ask_llm(context, question, sources):
    client = Client()

    system_message = """
You are a disciplined Swaminarayan scholar.

Rules:
- You may ONLY use the context and sources provided by the user.
- You MUST cite at least one Vachanamrut inline using the format: “as stated in Gadhadā I-11”.
- DO NOT reference any scripture outside the provided context.
- DO NOT use bracketed references like [1], [Gadhadā I-11], or fake labels like Sahasra or Tetrad.
- DO NOT invent verses or make up scripture names.
- Your tone should be spiritually uplifting, respectful, and clear — like a pravachan.

Never break these rules.
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

# Streamlit layout
st.set_page_config(page_title="Vachanamrut GPT", layout="centered")
st.markdown("<h1 style='text-align: center;'>🕉️ Vachanamrut GPT</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask spiritually grounded questions from the Vachanamrut, with wisdom from Bhagvān</p>", unsafe_allow_html=True)

query = st.text_input("🙏 Ask a question about the Vachanamrut:")

if query:
    top_k = 5
    similarity_threshold = 0.60
    query_embedding = model.encode(query).astype("float32").reshape(1, -1)
    distances, indices_found = index.search(query_embedding, top_k)

    all_context_chunks = []
    used_sources = set()

    for i, idx in enumerate(indices_found[0]):
        similarity_score = 1 / (1 + distances[0][i])
        if similarity_score < similarity_threshold:
            continue
        match_id = chunks[idx]["vachanamrut_id"]
        chunk_index = chunks[idx]["chunk_index"]
        source_title = f"{match_id} – {chunks[idx]['title']}"
        used_sources.add(source_title)
        for offset in [-2, -1, 0, 1, 2]:
            neighbor_idx = chunk_index + offset
            for c in chunks:
                if c["vachanamrut_id"] == match_id and c["chunk_index"] == neighbor_idx:
                    all_context_chunks.append(c["text"])

    full_context = " ".join(all_context_chunks).replace("\n", " ").strip()
    full_context = re.sub(r"\s{2,}", " ", full_context)

    with st.spinner("✨ Thinking deeply..."):
        answer = ask_llm(full_context, query, sorted(used_sources))

    if any(term in answer.lower() for term in ["sahasra", "[", "]", "tetrad"]):
        st.error("⚠️ This response included a forbidden citation style. Please rephrase or try again.")
    else:
        st.markdown("---")
        st.markdown(f"**📖 Question:** {query}")
        st.markdown("**🧘🏽‍♂️ Answer:**")
        st.markdown(f"<div style='background-color:#f9f9f9;padding:20px;border-radius:10px'>{answer}</div>", unsafe_allow_html=True)