import faiss

# Load precomputed embeddings
recipe_vectors = np.load("recipe_embeddings.npy")

# Create FAISS index
index = faiss.IndexFlatL2(recipe_vectors.shape[1])
index.add(recipe_vectors)

# Save the FAISS index
faiss.write_index(index, "recipe_index.faiss")
