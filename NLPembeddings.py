import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load dataset
with open("ingr_map.pkl", "rb") as f:
    ingr_map = pickle.load(f)

# Extract processed ingredient names
ingredient_list = ingr_map["replaced"]

# Load preprocessed recipes
recipes = pd.read_csv("PP_recipes.csv")

# Use a transformer model to generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert each recipe's ingredients into embeddings
recipe_embeddings = model.encode(recipes["ingredients"].astype(str), convert_to_numpy=True)

# Save embeddings for later use
np.save("recipe_embeddings.npy", recipe_embeddings)
recipes["embedding_index"] = range(len(recipes))  # Add index mapping
recipes.to_csv("processed_recipes.csv", index=False)
