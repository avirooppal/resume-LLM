# Resume Parser and Job Matcher

A powerful resume parsing and job matching system that uses advanced NLP and machine learning techniques to analyze resumes and match them with job descriptions. The system provides accurate parsing of resumes in various formats (PDF, DOCX) and intelligent matching against job requirements.

## Model Stack

### Core Components

1. **Resume Parser**
   - Document parsing for PDF and DOCX formats
   - Structured data extraction for:
     - Personal information
     - Work experience
     - Education
     - Skills
     - Projects
   - Experience calculation and validation

2. **Job Description Analyzer**
   - Requirement extraction
   - Skill and qualification parsing
   - Experience level detection

3. **Matching Engine**
   - Semantic similarity scoring using `sentence-transformers/all-MiniLM-L6-v2`
   - Skill matching with normalization
   - Experience validation
   - Education verification
   - Weighted scoring system

### Key Technologies

- **Deep Learning**: 
  - PyTorch 2.1.0
  - Transformers 4.30.0
  - Sentence Transformers (all-MiniLM-L6-v2)
- **Machine Learning**: scikit-learn 1.3.2
- **Document Processing**: 
  - PyPDF2 3.0.1
  - python-docx 1.0.1
- **API Framework**: 
  - FastAPI 0.104.1
  - uvicorn 0.24.0
- **Data Validation**: pydantic 1.10.13
- **Data Processing**: numpy 1.24.3

### Model Details

The system uses the following pre-trained models:

1. **Text Embeddings**:
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Purpose: Semantic text understanding and similarity scoring
   - Features:
     - 384-dimensional embeddings
     - Optimized for semantic similarity tasks
     - Fast inference with minimal resource requirements

2. **Text Processing**:
   - Tokenizer: AutoTokenizer from Hugging Face
   - Model: AutoModel from Hugging Face
   - Purpose: Text tokenization and feature extraction

## Features

- Multi-format resume parsing (PDF, DOCX)
- Accurate information extraction
- Skill normalization and matching
- Experience calculation
- Education verification
- Semantic matching with job descriptions
- Detailed match scoring and analysis
- RESTful API interface

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the API server:
```bash
uvicorn api:app --reload
```

## API Usage

The API provides the following endpoints:

- `POST /api/parse-resume`: Parse a resume and return structured data
- `POST /api/match-job`: Match a resume against a job description
- `GET /api/health`: Health check endpoint

Example request:
```python
import requests

# Parse resume
response = requests.post(
    "http://localhost:8000/api/parse-resume",
    files={"file": open("resume.pdf", "rb")}
)

# Match with job description
response = requests.post(
    "http://localhost:8000/api/match-job",
    json={
        "resume_data": resume_data,
        "job_description": "Your job description here"
    }
)
```

## Project Structure

```
resume/
├── api.py              # FastAPI application and endpoints
├── main.py            # Core matching and parsing logic
├── modules/           # Modular components
├── utils/            # Utility functions
├── models/           # ML models and embeddings
├── config/           # Configuration files
├── data/             # Data storage
├── files/            # File processing utilities
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

## Dependencies

- torch==2.1.0
- transformers==4.30.0
- numpy==1.24.3
- scikit-learn==1.3.2
- fastapi==0.104.1
- uvicorn==0.24.0
- python-multipart==0.0.6
- pydantic==1.10.13
- PyPDF2==3.0.1
- python-docx==1.0.1

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License
