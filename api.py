from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from modules.preprocessing import ResumePreprocessor, JDPreprocessor
from modules.feature_extraction import FeatureExtractor
from modules.matching import JDMatcher, MatchAnalyzer
import PyPDF2
import json

app = FastAPI(title="Resume-JD Matching API")

# Initialize components
resume_pp = ResumePreprocessor()
jd_pp = JDPreprocessor()
feature_extractor = FeatureExtractor()
matcher = JDMatcher()
match_analyzer = MatchAnalyzer()

def extract_text_from_file(upload_file: UploadFile) -> str:
    if upload_file.filename.lower().endswith('.pdf'):
        try:
            reader = PyPDF2.PdfReader(upload_file.file)
            return ''.join(page.extract_text() or '' for page in reader.pages)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading PDF: {str(e)}")
    elif upload_file.filename.lower().endswith('.txt'):
        return upload_file.file.read().decode('utf-8')
    else:
        raise HTTPException(status_code=415, detail="Unsupported file format. Use .txt or .pdf")

@app.post("/add_jd/")
async def add_jd(jd_file: UploadFile = File(...)):
    jd_text = extract_text_from_file(jd_file)

    if jd_pp.validate_jd(jd_text):
        clean_jd = jd_pp.clean_jd(jd_text)
        jd_embedding = feature_extractor.get_embeddings([clean_jd])
        if jd_embedding:
            jd_meta = {
                "text": clean_jd,
                "requirements": jd_pp.extract_requirements(clean_jd)
            }
            matcher.add_jd(jd_embedding, jd_meta)
            return {"message": "JD successfully added to the database"}
        else:
            raise HTTPException(status_code=500, detail="JD embedding generation failed")
    else:
        raise HTTPException(status_code=400, detail="Invalid JD text")

@app.post("/match/")
async def match_resume_to_jd(
    resume_file: UploadFile = File(...),
    jd_file: UploadFile = File(...)
):
    resume_text = extract_text_from_file(resume_file)
    jd_text = extract_text_from_file(jd_file)

    if not resume_pp.validate_input(resume_text):
        raise HTTPException(status_code=400, detail="Resume validation failed")

    if not jd_pp.validate_jd(jd_text):
        raise HTTPException(status_code=400, detail="JD validation failed")

    clean_resume = resume_pp.clean_text(resume_text)
    clean_jd = jd_pp.clean_text(jd_text)

    resume_embedding = feature_extractor.get_embeddings([clean_resume])
    jd_embedding = feature_extractor.get_embeddings([clean_jd])

    if resume_embedding is None or jd_embedding is None:
        raise HTTPException(status_code=500, detail="Embedding generation failed")

    temp_matcher = JDMatcher()
    temp_matcher.add_jd(jd_embedding, {"text": clean_jd})
    matches = temp_matcher.find_matches(resume_embedding)

    if not matches:
        return {"message": "No suitable match found", "semantic_score": 0}

    resume_sections = resume_pp.extract_sections(clean_resume)
    jd_requirements = jd_pp.extract_requirements(clean_jd)
    semantic_score = matches[0]['score']

    match_results = match_analyzer.analyze_match(
        resume_data=resume_sections,
        jd_data={"requirements": jd_requirements},
        semantic_score=semantic_score
    )

    return JSONResponse(content={
        "semantic_score": semantic_score,
        "resume_sections": resume_sections,
        "jd_requirements": jd_requirements,
        "match_results": json.loads(match_analyzer.to_json(match_results))
    })
