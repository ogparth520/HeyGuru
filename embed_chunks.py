import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Load chunked Vachanamrut data
with open("vachanamrut_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed with a progress bar
for chunk in tqdm(chunks, desc="🔄 Embedding chunks"):
    chunk["embedding"] = model.encode(chunk["text"]).tolist()

# Save new file
with open("vachanamrut_chunks_with_embeddings.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"\n✅ Saved {len(chunks)} embedded chunks to vachanamrut_chunks_with_embeddings.json")