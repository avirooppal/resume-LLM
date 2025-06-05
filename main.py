import argparse
from modules.preprocessing import ResumePreprocessor, JDPreprocessor
from modules.feature_extraction import FeatureExtractor
from modules.matching import JDMatcher, MatchAnalyzer
from config.settings import settings
import numpy as np
import PyPDF2
import json

def main():
    parser = argparse.ArgumentParser(description="Resume-JD Matching System")
    parser.add_argument("--resume", help="Path to resume text file")
    parser.add_argument("--jd", help="Path to job description text file")
    parser.add_argument("--add_jd", action="store_true", help="Add JD to database")
    args = parser.parse_args()

    # Initialize components
    resume_pp = ResumePreprocessor()
    jd_pp = JDPreprocessor()
    feature_extractor = FeatureExtractor()
    matcher = JDMatcher()
    match_analyzer = MatchAnalyzer()

    if args.add_jd and args.jd:
        # Add new JD to the system
        with open(args.jd, "r") as f:
            jd_text = f.read()
        
        if jd_pp.validate_jd(jd_text):
            clean_jd = jd_pp.clean_jd(jd_text)
            jd_embedding = feature_extractor.get_embeddings([clean_jd])
            
            if jd_embedding is not None:
                jd_meta = {
                    "text": clean_jd,
                    "requirements": jd_pp.extract_requirements(clean_jd)
                }
                matcher.add_jd(jd_embedding, jd_meta)
                print("Successfully added JD to database")
            else:
                print("Failed to generate JD embeddings")
        else:
            print("Invalid JD text")

    elif args.resume and args.jd:
        # Match resume against existing JDs
        resume_path = args.resume
        if resume_path.lower().endswith('.pdf'):
            try:
                with open(resume_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    resume_text = ''
                    for page_num in range(len(reader.pages)):
                        resume_text += reader.pages[page_num].extract_text()
            except Exception as e:
                print(f"Error reading PDF file: {e}")
                return # Exit if PDF reading fails
        elif resume_path.lower().endswith('.txt'):
            with open(resume_path, "r", encoding='utf-8') as f:
                resume_text = f.read()
        else:
            print("Unsupported resume file format. Please provide a .txt or .pdf file.")
            return # Exit if file format is unsupported
        
        with open(args.jd, "r") as f:
            jd_text = f.read()

        if resume_pp.validate_input(resume_text) and jd_pp.validate_jd(jd_text):
            clean_resume = resume_pp.clean_text(resume_text)
            clean_jd = jd_pp.clean_jd(jd_text)
            
            print("\nCleaned Resume Text:")
            print("-" * 50)
            print(clean_resume[:500] + "..." if len(clean_resume) > 500 else clean_resume)
            print("-" * 50)
            
            print("\nCleaned JD Text:")
            print("-" * 50)
            print(clean_jd[:500] + "..." if len(clean_jd) > 500 else clean_jd)
            print("-" * 50)
            
            # Get embeddings
            resume_embedding = feature_extractor.get_embeddings([clean_resume])
            jd_embedding = feature_extractor.get_embeddings([clean_jd])
            
            if resume_embedding is not None and jd_embedding is not None:
                # Add JD temporarily for this match
                temp_matcher = JDMatcher()
                temp_matcher.add_jd(jd_embedding, {"text": clean_jd})
                
                # Find matches
                matches = temp_matcher.find_matches(resume_embedding)
                
                if matches:
                    # Extract resume sections
                    resume_sections = resume_pp.extract_sections(clean_resume)
                    
                    # Extract JD requirements
                    jd_requirements = jd_pp.extract_requirements(clean_jd)
                    
                    # Calculate semantic similarity score
                    semantic_score = matches[0]['score']
                    
                    print("\nSemantic Score:", semantic_score)
                    print("\nResume Sections:")
                    print(json.dumps(resume_sections, indent=2))
                    print("\nJD Requirements:")
                    print(json.dumps(jd_requirements, indent=2))
                    
                    # Analyze match
                    match_results = match_analyzer.analyze_match(
                        resume_data=resume_sections,
                        jd_data={"requirements": jd_requirements},
                        semantic_score=semantic_score
                    )
                    
                    # Output JSON results
                    print("\nMatch Results:")
                    print(match_analyzer.to_json(match_results))
                else:
                    print("No matches found - Semantic similarity too low")
                    print("Try adjusting the similarity threshold in JDMatcher.find_matches()")
            else:
                print("Failed to generate embeddings")
                if resume_embedding is None:
                    print("Resume embedding generation failed")
                if jd_embedding is None:
                    print("JD embedding generation failed")
        else:
            print("Invalid resume or JD text")
            if not resume_pp.validate_input(resume_text):
                print("Resume validation failed")
            if not jd_pp.validate_jd(jd_text):
                print("JD validation failed")
    else:
        print("Please provide valid arguments")

if __name__ == "__main__":
    main()