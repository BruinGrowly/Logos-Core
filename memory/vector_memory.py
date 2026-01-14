import numpy as np
from sentence_transformers import SentenceTransformer

class VectorMemory:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"[Memory] Initializing Vector Memory with model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
            print("[Memory] Model loaded successfully.")
        except Exception as e:
            print(f"[Memory] Failed to load model: {e}")
            self.model = None

    def encode(self, text):
        """
        Encodes a text into a vector.
        """
        if self.model is None:
            return None
        return self.model.encode(text)

    def calculate_similarity(self, vector_a, vector_b):
        """
        Calculates cosine similarity between two vectors.
        """
        if vector_a is None or vector_b is None:
            return 0.0
        
        # Ensure vectors are numpy arrays
        vec_a = np.array(vector_a)
        vec_b = np.array(vector_b)

        # Normalize vectors
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return np.dot(vec_a, vec_b) / (norm_a * norm_b)
