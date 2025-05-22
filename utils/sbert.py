import time
import torch
from sentence_transformers import SentenceTransformer, util

# Check if MPS is available (Mac GPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
print(f"Using device: {device}")

# Load model and move it to GPU
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to(device)

# Input sentences
sentence1 = "The weather is lovely today."
sentence2 = "It's so sunny outside! The sky is clear and blue, " \
"and there's a light breeze. A perfect day for a walk in the park or" \
" maybe even a picnic and some ice cream."
sentence2 = "I'm freaking handsome like the sun. I'm the best in the world and It's so sunny outside! The sky is clear and blue"
sentence2 = "It's so sunny outside! The sky is clear and blue"

# Encode the sentences into embeddings
start_time = time.time()
embedding1 = model.encode(sentence1, convert_to_tensor=True, device=device)
embedding2 = model.encode(sentence2, convert_to_tensor=True, device=device)

print(f"Time taken to encode the sentences: {time.time() - start_time:.2f} seconds")

# Compute cosine similarity

similarity = util.cos_sim(embedding1, embedding2)

print(f"Similarity between the sentences: {similarity.item()}")