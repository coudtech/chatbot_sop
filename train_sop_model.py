import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util

# Load SOP Excel
df = pd.read_excel("sop_docs/sop_data3.xlsx")

# Create embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# For each SOP Content, generate embeddings
df['embedding'] = df['Content'].apply(lambda x: embed_model.encode(x, convert_to_tensor=True))

# Save both dataframe & embedding model
with open("sop_embeddings.pkl", "wb") as f:
    pickle.dump(df, f)

# Save embedding model separately for inference
embed_model.save("embed_model")
print("Smart SOP model and embeddings saved!")
