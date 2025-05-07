import json

# Load the original structured Vachanamrut
with open("vachanamrut_structured.json", "r", encoding="utf-8") as f:
    vachs = json.load(f)

chunks = []

for v in vachs:
    paragraphs = [p.strip() for p in v["content"].split("\n") if p.strip()]
    for i, para in enumerate(paragraphs):
        chunks.append({
            "vachanamrut_id": v["id"],
            "title": v["title"],
            "chunk_index": i,
            "text": para
        })

# Save to file
with open("vachanamrut_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Created {len(chunks)} chunks and saved to vachanamrut_chunks.json")