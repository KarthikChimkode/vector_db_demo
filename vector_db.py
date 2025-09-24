from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

text = [
    "I love machine learning",
    "Deep learning is a subset of machine learning",
    "Bananas are yellow",
    "Apples are red and sweet",
    "I enjoy studying artificial intelliegence",
]

model =  SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(text)
print("Shape:", embeddings.shape)
d = embeddings.shape[1]


index = faiss.IndexFlatL2(d)

index.add(np.array(embeddings))
print("Stored vectors:", index.ntotal)

query = "Ballu said who are you?"
query_vec = model.encode([query])

k = 2
D, I =index.search(query_vec, k)

print("\nQuery:", query)
print("\nTop matches:")
for idx, dist in zip(I[0], D[0]):
    print(f"Text: {text[idx]} (distance = {dist:.4f})")


