import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Union

class EmbeddingGenerator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator with a transformer model.
        
        Args:
            model_name (str): Name of the transformer model to use.
                             Default is "sentence-transformers/all-MiniLM-L6-v2"
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        self.model = AutoModel.from_pretrained(model_name, local_files_only=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text (str): Input text to generate embedding for.
            
        Returns:
            np.ndarray: Embedding vector for the input text.
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            
        return embeddings.cpu().numpy()
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts (List[str]): List of input texts to generate embeddings for.
            
        Returns:
            np.ndarray: Array of embedding vectors for the input texts.
        """
        # Tokenize and prepare input
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            
        return embeddings.cpu().numpy()
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on the model output.
        
        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask for the input
            
        Returns:
            torch.Tensor: Mean pooled embeddings
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 0) / torch.clamp(input_mask_expanded.sum(0), min=1e-9)
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding vector.
            embedding2 (np.ndarray): Second embedding vector.
            
        Returns:
            float: Cosine similarity score between the embeddings.
        """
        # Normalize the embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2.T) / (norm1 * norm2)
        return float(similarity) 