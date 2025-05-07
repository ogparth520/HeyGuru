import json
import numpy as np
import faiss

# Load your data
with open("vachanamrut_with_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract embeddings and metadata
embeddings = np.array([entry["embedding"] for entry in data]).astype("float32")
texts = [entry["content"] for entry in data]
titles = [entry["title"] for entry in data]
ids = [entry["id"] for entry in data]

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Function to search
def search_vachanamrut(query, k=3):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, k)

    print(f"\n🔍 Top {k} matches for: \"{query}\"\n")
    for i in indices[0]:
        print(f"📖 {ids[i]} – {titles[i]}\n{texts[i][:500]}...\n")

# Ask a question
if __name__ == "__main__":
    user_q = input("Ask a question about the Vachanamrut:\n> ")
    search_vachanamrut(user_q)