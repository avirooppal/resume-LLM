class Settings:
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    MAX_TEXT_LENGTH = 512  # For tokenizer
    
    # FAISS settings
    FAISS_INDEX_TYPE = "FlatL2"
    
    # Resource constraints
    MAX_MEMORY_USAGE = 1.8  # GB
    
    # Validation thresholds
    MIN_RESUME_LENGTH = 50
    MIN_JD_LENGTH = 20

settings = Settings()