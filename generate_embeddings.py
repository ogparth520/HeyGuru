import json
from sentence_transformers import SentenceTransformer
import numpy as np

# ✅ Load the structured Vachanamrut JSON
with open("vachanamrut_structured.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ✅ Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Generate and add embeddings
for entry in data:
    embedding = model.encode(entry["content"])
    entry["embedding"] = embedding.tolist()

# ✅ Save to new file
with open("vachanamrut_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ Embeddings saved to vachanamrut_with_embeddings.json")