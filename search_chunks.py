import json
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from ollama import Client

# Load Vachanamrut chunk data with embeddings
with open("vachanamrut_chunks_with_embeddings.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Prepare FAISS index and metadata
embeddings = np.array([chunk["embedding"] for chunk in chunks]).astype("float32")
ids = [chunk["vachanamrut_id"] for chunk in chunks]
titles = [chunk["title"] for chunk in chunks]
indices = [chunk["chunk_index"] for chunk in chunks]

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# LLM function using Ollama
def ask_llm(context, question, sources):
    client = Client()

    prompt = f"""
You are a spiritually inspiring Swaminarayan scholar who teaches through clarity and warmth.
You must only use the Vachanamrut for your answers, based on the context provided.
You speak in a reverent, flowing tone, like a guru giving pravachan.

Instructions:
- Use terminology from the Vachanamrut (e.g., māyā, antahkaran, Bhagvān, vrutti, mukti).
- You MUST cite at least one of the provided sources inline using the format: “as explained in Gadhadā I-11”.
- Do NOT use bracketed citations like [1], [2], or any numbered references.
- Do NOT invent scripture titles (e.g., 'Sahisra 702', 'Tetrad 6').
- You are not allowed to reference any text outside of the Vachanamrut.
- Focus on spiritual meaning, transformation, and guidance, not academic tone.

Sources you are allowed to reference:
{chr(10).join(f"- {s}" for s in sources)}

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']

# Main conversational loop
if __name__ == "__main__":
    top_k = 5
    similarity_threshold = 0.60

    while True:
        query = input("\nAsk a question about the Vachanamrut:\n> ")
        if query.lower() in ["exit", "quit", "no"]:
            print("\n🙏 Jai Swaminarayan. May your understanding grow through the words of Bhagvān.\n")
            break

        # Embed the query and search index
        query_embedding = model.encode(query).astype("float32").reshape(1, -1)
        distances, indices_found = index.search(query_embedding, top_k)

        all_context_chunks = []
        used_sources = set()

        for i, idx in enumerate(indices_found[0]):
            similarity_score = 1 / (1 + distances[0][i])
            if similarity_score < similarity_threshold:
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

        print("\n🤖 LLM Answer:\n")
        answer = ask_llm(full_context, query, sorted(used_sources))
        print(answer)

        follow_up = input("\n💬 Would you like to ask another question? (yes/no):\n> ")
        if follow_up.lower() not in ["yes", "y"]:
            print("\n🙏 Jai Swaminarayan. May your understanding grow through the words of Bhagvān.\n")
            break