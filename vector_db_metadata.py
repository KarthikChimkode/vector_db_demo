import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

texts = [
    "I love machine learning",
    "Deep learning is a subset of machine learning",
    "Bananas are yellow",
    "Apples are red and sweet",
    "I enjoy studying artificial intelligence",
]

conn = sqlite3.connect("metadata.db")
c = conn.cursor()


c.execute("DROP TABLE IF EXISTS docs")
c.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, text TEXT)")

c.executemany(
    'INSERT OR REPLACE INTO docs (id, text) VALUES (?, ?)',
    [(i,t) for i, t in enumerate(texts)]

)
conn.commit()

model = SentenceTransformer("all-MiniLM-L6-v2")

c.execute("SELECT id, text From docs")
rows = c.fetchall()
ids, texts_from_db = zip(*rows)

embeddings = model.encode(list(texts_from_db), convert_to_numpy=True).astype("float32")

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

print("Stored vectors:", index.ntotal)

query = "Karthik 1"
query_vec = model.encode([query], convert_to_numpy=True).astype("float32")


k = 2
D, I = index.search(query_vec, k)

print("\nQuery:", query)
print("Top matchjes")
for idx, dist in zip(I[0], D[0]):
    print(I[0], D[0])
    c.execute("SELECT text FROM docs WHERE id=?", (int(idx),))
    print(f"Text: {c.fetchone()[0]} (distance = {dist:.4f})")

conn.close()