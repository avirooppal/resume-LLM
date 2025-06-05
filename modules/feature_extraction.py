from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List, Optional
from config.settings import settings
import warnings
import gc

class FeatureExtractor:
    def __init__(self):
        # Initialize with small LLM
        self.model_name = settings.EMBEDDING_MODEL
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir="models/pretrained"
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir="models/pretrained"
        )
        
        # Optimize for CPU and low memory
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
    def __del__(self):
        """Clean up to free memory"""
        del self.model
        del self.tokenizer
        gc.collect()
        
    def get_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate embeddings for text using MiniLM"""
        if not texts or not all(isinstance(t, str) for t in texts):
            warnings.warn("Invalid input texts for embedding")
            return None
            
        try:
            # Tokenize with truncation
            inputs = self.tokenizer(
                texts, 
                padding=True,
                truncation=True,
                max_length=settings.MAX_TEXT_LENGTH,
                return_tensors="pt"
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Mean pooling
            embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
            
            # Convert to numpy and free memory
            embeddings_np = embeddings.numpy()
            del inputs, outputs, embeddings
            gc.collect()
            
            return embeddings_np
            
        except Exception as e:
            warnings.warn(f"Embedding generation failed: {str(e)}")
            return None
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        return torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)