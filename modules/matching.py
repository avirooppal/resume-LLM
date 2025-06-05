import faiss
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os
import pickle
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class MatchDetails:
    semantic_score: float
    skill_score: float
    matched_skills: List[str]
    missing_skills: List[str]
    semantically_matched_skills: List[Tuple[str, str, float]]
    experience_score: float
    education_score: float
    calculated_resume_experience_years: float
    weights: Dict[str, float]
    basics: Dict[str, Any] = None
    work: List[Dict[str, Any]] = None
    education: List[Dict[str, Any]] = None
    publications: List[Dict[str, Any]] = None
    skills: List[Dict[str, Any]] = None
    languages: List[Dict[str, Any]] = None
    projects: List[Dict[str, Any]] = None
    certificates: List[Dict[str, Any]] = None

@dataclass
class MatchResults:
    match_percentage: float
    details: MatchDetails
    parsed_resume: Dict[str, Any] = None

class ResumeDataCleaner:
    """Clean and standardize resume data for better matching"""
    
    @staticmethod
    def clean_basics(basics: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and extract proper basic information"""
        cleaned = {}
        
        # Extract name - use the first proper name found
        name = basics.get("name", "")
        if not name or len(name) > 100:  # If name is missing or too long
            # Try to extract from label or summary
            text = basics.get("label", "") + " " + basics.get("summary", "")
            name_match = re.search(r'^([A-Z][a-z]+ [A-Z][a-z]+)', text)
            if name_match:
                name = name_match.group(1)
        cleaned["name"] = name.strip()
        
        # Extract email with better pattern
        email = basics.get("email", "")
        text_to_search = str(basics)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, text_to_search)
        cleaned["email"] = email_match.group() if email_match else email.strip()
            
        # Extract phone with better pattern
        phone = basics.get("phone", "")
        phone_pattern = r'(?:\+?91[-.\s]?)?[6-9]\d{9}'
        phone_match = re.search(phone_pattern, text_to_search)
        if phone_match:
            cleaned["phone"] = phone_match.group()
        else:
            # Try to extract from phone field
            phone_digits = re.findall(r'\d+', phone)
            if phone_digits:
                cleaned["phone"] = "-".join(phone_digits[:3])
            else:
                cleaned["phone"] = phone.strip()
            
        # Clean summary
        summary = basics.get("summary", "")
        if len(summary) > 500:  # If too long, extract meaningful part
            sentences = re.split(r'[.!?]', summary)
            summary = '. '.join(s.strip() for s in sentences[:3] if s.strip()) + '.'
        cleaned["summary"] = summary.strip()
        
        # Clean location
        location = basics.get("location", {})
        if isinstance(location, str):
            # Parse string location
            city = re.search(r'\b(Bangalore|Mumbai|Delhi|Hyderabad|Chennai)\b', location, re.IGNORECASE)
            if city:
                location = {"city": city.group(1)}
        cleaned["location"] = location
        
        # Clean profiles
        profiles = basics.get("profiles", [])
        if isinstance(profiles, str):
            # Extract LinkedIn URL
            linkedin = re.search(r'(https?://(?:www\.)?linkedin\.com/[^\s]+)', profiles)
            profiles = [{"network": "LinkedIn", "url": linkedin.group(1)}] if linkedin else []
        cleaned["profiles"] = profiles
        
        return cleaned
    
    @staticmethod
    def extract_skills_from_text(text: str) -> List[str]:
        """Extract skills from any text using common patterns"""
        # Common technical skills patterns
        skill_patterns = [
            r'\b(?:Java|Python|JavaScript|C\+\+|C#|PHP|Ruby|Go|Rust|Swift|Kotlin)\b',
            r'\b(?:React|Angular|Vue|Node\.js|Express|Django|Flask|Spring Boot?|Laravel)\b',
            r'\b(?:SQL|MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|Oracle)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git|CI/CD|DevOps)\b',
            r'\b(?:HTML|CSS|Bootstrap|Tailwind|SASS|LESS)\b',
            r'\b(?:REST(?:ful)?|GraphQL|API|Microservices|Agile|Scrum)\b'
        ]
        
        skills = []
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)
        
        # Also look for skills after common keywords
        skill_keywords = ['skills:', 'technologies:', 'tools:', 'languages:', 'expertise:']
        for keyword in skill_keywords:
            if keyword.lower() in text.lower():
                start_idx = text.lower().find(keyword.lower())
                # Extract next 200 characters and look for comma-separated skills
                skill_section = text[start_idx:start_idx+200]
                potential_skills = re.findall(r'\b[A-Z][A-Za-z0-9+#.]*(?:\s+[A-Z][A-Za-z0-9+#.]*)*\b', skill_section)
                skills.extend(skill[:10] for skill in potential_skills if len(skill) > 1)
        
        return list(set([s.strip() for s in skills if s.strip()]))  # Remove duplicates and empty strings

    @staticmethod
    def clean_work_experience(work: List[Dict[str, Any]], text_content: str = "") -> List[Dict[str, Any]]:
        """Clean and extract proper work experience"""
        cleaned_work = []
        
        # If work array is empty or malformed, try to extract from text
        if not work or any(not isinstance(w, dict) for w in work):
            # Enhanced patterns for work experience extraction
            lines = text_content.split('\n')
            current_job = None
            in_experience_section = False
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Check if we're in experience section
                if re.match(r'^EXPERIENCE|^WORK EXPERIENCE|^EMPLOYMENT|^PROFESSIONAL EXPERIENCE', line, re.IGNORECASE):
                    in_experience_section = True
                    continue
                
                # Stop if we hit another major section
                if re.match(r'^(EDUCATION|SKILLS|PROJECTS|CERTIFICATIONS|TECHNICAL SKILLS)', line, re.IGNORECASE):
                    if current_job:
                        cleaned_work.append(current_job)
                    break
                
                if in_experience_section or any(role in line.lower() for role in ['engineer', 'developer', 'analyst', 'manager']):
                    # Look for job title and company pattern: "Position | Company | Location | Date"
                    if '|' in line and any(word in line.lower() for word in ['engineer', 'developer', 'analyst', 'manager']):
                        if current_job:
                            cleaned_work.append(current_job)
                        
                        parts = [p.strip() for p in line.split('|')]
                        current_job = {
                            "company": parts[1] if len(parts) > 1 else "",
                            "position": parts[0] if len(parts) > 0 else "",
                            "startDate": "",
                            "endDate": "",
                            "highlights": []
                        }
                        
                        # Extract dates from the line or next lines
                        date_pattern = r'(\w{3}\s+\d{4})\s*[-–]\s*(\w{3}\s+\d{4}|\w+\s+\d{4}|Present|Current)'
                        date_match = re.search(date_pattern, line)
                        if not date_match and i + 1 < len(lines):
                            date_match = re.search(date_pattern, lines[i + 1])
                        
                        if date_match:
                            try:
                                start_str = date_match.group(1)
                                end_str = date_match.group(2)
                                
                                # Convert to standard format
                                start_date = datetime.strptime(start_str, '%b %Y').strftime('%Y-%m-%d')
                                if end_str.lower() in ['present', 'current']:
                                    end_date = datetime.now().strftime('%Y-%m-%d')
                                else:
                                    end_date = datetime.strptime(end_str, '%b %Y').strftime('%Y-%m-%d')
                                
                                current_job["startDate"] = start_date
                                current_job["endDate"] = end_date
                            except:
                                # Fallback to year extraction
                                years = re.findall(r'\d{4}', line)
                                if len(years) >= 2:
                                    current_job["startDate"] = f"{years[0]}-01-01"
                                    current_job["endDate"] = f"{years[1]}-12-31"
                    
                    # Look for bullet points or achievements
                    elif line.startswith('-') or line.startswith('•') or line.startswith('*'):
                        if current_job:
                            highlight = line[1:].strip() if line[0] in ['-', '•', '*'] else line
                            current_job["highlights"].append(highlight)
                    elif current_job and not current_job.get("company") and len(line.split()) < 5:
                        # Might be company name alone
                        current_job["company"] = line
            
            # Add the last job if exists
            if current_job:
                cleaned_work.append(current_job)
        else:
            # Clean existing work data
            for job in work:
                if not isinstance(job, dict):
                    continue
                    
                cleaned_job = {
                    "company": job.get("name", job.get("company", "")).strip(),
                    "position": job.get("position", "").strip(),
                    "startDate": job.get("startDate", "").strip(),
                    "endDate": job.get("endDate", "").strip(),
                    "highlights": [h.strip() for h in job.get("highlights", []) if h.strip()]
                }
                
                # Convert dates to standard format if needed
                try:
                    if cleaned_job["startDate"] and len(cleaned_job["startDate"].split()) == 2:
                        start_date = datetime.strptime(cleaned_job["startDate"], "%b %Y")
                        cleaned_job["startDate"] = start_date.strftime("%Y-%m-%d")
                    
                    if cleaned_job["endDate"]:
                        if cleaned_job["endDate"].lower() in ['present', 'current']:
                            cleaned_job["endDate"] = datetime.now().strftime("%Y-%m-%d")
                        elif len(cleaned_job["endDate"].split()) == 2:
                            end_date = datetime.strptime(cleaned_job["endDate"], "%b %Y")
                            cleaned_job["endDate"] = end_date.strftime("%Y-%m-%d")
                except Exception as e:
                    print(f"Error converting dates: {e}")
                
                cleaned_work.append(cleaned_job)
        
        return cleaned_work
    
    @staticmethod
    def clean_education(education: List[Dict[str, Any]], text_content: str = "") -> List[Dict[str, Any]]:
        """Clean and extract proper education information"""
        cleaned_edu = []
        
        # If education array is empty or malformed, try to extract from text
        if not education or any(not isinstance(e, dict) for e in education):
            lines = text_content.split('\n')
            in_education_section = False
            current_edu = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if we're in education section
                if re.match(r'^EDUCATION|^ACADEMIC BACKGROUND', line, re.IGNORECASE):
                    in_education_section = True
                    continue
                
                # Stop if we hit another major section
                if re.match(r'^(EXPERIENCE|SKILLS|PROJECTS)', line, re.IGNORECASE):
                    if current_edu:
                        cleaned_edu.append(current_edu)
                    break
                
                if in_education_section:
                    # Look for degree patterns
                    degree_pattern = r'(B\.?Tech|B\.?E\.?|Bachelor|M\.?Tech|M\.?E\.?|Master|PhD|Diploma)\b.*?\b(?:in\s+)?(Computer Science|Engineering|IT|Information Technology)?'
                    match = re.search(degree_pattern, line, re.IGNORECASE)
                    if match:
                        if current_edu:
                            cleaned_edu.append(current_edu)
                        
                        current_edu = {
                            "institution": "",
                            "area": match.group(2) or "Computer Science",
                            "studyType": match.group(1),
                            "startDate": "",
                            "endDate": ""
                        }
                        
                        # Extract institution (usually follows degree)
                        institution = line[match.end():].split('|')[0].strip()
                        if institution:
                            current_edu["institution"] = institution
                        
                        # Extract years
                        years = re.findall(r'\d{4}', line)
                        if len(years) >= 2:
                            current_edu["startDate"] = f"{years[0]}-01-01"
                            current_edu["endDate"] = f"{years[1]}-12-31"
                        elif len(years) == 1:
                            current_edu["endDate"] = f"{years[0]}-12-31"
            
            if current_edu:
                cleaned_edu.append(current_edu)
        else:
            # Clean existing education data
            for edu in education:
                if not isinstance(edu, dict):
                    continue
                
                # Split area and institution if combined
                area = edu.get("area", "")
                institution = edu.get("institution", "")
                
                if not institution and area:
                    # Try different separators
                    if "  " in area:  # Double space
                        parts = area.split("  ")
                        area = parts[0].strip()
                        institution = parts[1].strip()
                    elif "|" in area:  # Pipe
                        parts = area.split("|")
                        area = parts[0].strip()
                        institution = parts[1].strip()
                    elif "-" in area:  # Dash
                        parts = area.split("-")
                        area = parts[0].strip()
                        institution = parts[1].strip()
                
                cleaned_edu.append({
                    "institution": institution,
                    "area": area,
                    "studyType": edu.get("studyType", "").strip(),
                    "startDate": edu.get("startDate", "").strip(),
                    "endDate": edu.get("endDate", "").strip()
                })
        
        return cleaned_edu
    
    @staticmethod
    def extract_skills(resume_data: Dict[str, Any], text_content: str = "") -> List[str]:
        """Extract skills from resume data and text content"""
        skills = []
        
        # From structured skills data
        if isinstance(resume_data.get("skills"), list):
            for skill in resume_data["skills"]:
                if isinstance(skill, dict):
                    if skill.get("keywords"):
                        skills.extend(skill["keywords"])
                    if skill.get("name"):
                        skills.append(skill["name"])
                else:
                    skills.append(str(skill))
        
        # From work experience highlights
        if isinstance(resume_data.get("work"), list):
            for job in resume_data["work"]:
                if isinstance(job, dict) and isinstance(job.get("highlights"), list):
                    for highlight in job["highlights"]:
                        skills.extend(ResumeDataCleaner.extract_skills_from_text(str(highlight)))
        
        # From summary
        if isinstance(resume_data.get("basics"), dict) and resume_data["basics"].get("summary"):
            skills.extend(ResumeDataCleaner.extract_skills_from_text(resume_data["basics"]["summary"]))
        
        # From text content
        if text_content:
            skills.extend(ResumeDataCleaner.extract_skills_from_text(text_content))
        
        # Clean and deduplicate
        cleaned_skills = []
        seen_skills = set()
        for skill in skills:
            skill = str(skill).strip()
            if not skill or len(skill) <= 1:
                continue
                
            skill_lower = skill.lower()
            
            # Normalize REST-related skills
            if skill_lower in ['rest', 'restful', 'rest api', 'rest apis', 'restful api', 'restful apis']:
                if 'rest' not in seen_skills:
                    cleaned_skills.append("REST APIs")
                    seen_skills.add('rest')
            # Normalize CI/CD-related skills
            elif skill_lower in ['ci/cd', 'cicd', 'ci-cd', 'devops']:
                if 'ci/cd' not in seen_skills:
                    cleaned_skills.append("CI/CD")
                    seen_skills.add('ci/cd')
            # Normalize database-related skills
            elif skill_lower in ['postgresql', 'postgres db', 'postgresql db']:
                if 'postgres' not in seen_skills:
                    cleaned_skills.append("PostgreSQL")
                    seen_skills.add('postgres')
            elif skill_lower in ['mysql db', 'mysql database']:
                if 'mysql' not in seen_skills:
                    cleaned_skills.append("MySQL")
                    seen_skills.add('mysql')
            elif skill_lower in ['mongodb', 'mongo', 'mongo db', 'mongodb database']:
                if 'mongodb' not in seen_skills:
                    cleaned_skills.append("MongoDB")
                    seen_skills.add('mongodb')
            # Handle other skills
            elif skill_lower not in seen_skills:
                cleaned_skills.append(skill)
                seen_skills.add(skill_lower)
        
        return cleaned_skills

class JDMatcher:
    def __init__(self):
        self.jds = []
        self.similarity_threshold = 0.3  # Lower threshold for better matching
        
    def add_jd(self, embedding, metadata):
        self.jds.append({
            "embedding": embedding,
            "metadata": metadata
        })
        
    def find_matches(self, resume_embedding):
        if not self.jds:
            return []
            
        matches = []
        for jd in self.jds:
            similarity = cosine_similarity(resume_embedding, jd["embedding"])[0][0]
            if similarity >= self.similarity_threshold:
                matches.append({
                    "score": float(similarity),
                    "metadata": jd["metadata"]
                })
                
        # Sort matches by score in descending order
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches

class MatchAnalyzer:
    def __init__(self):
        self.weights = {
            "semantic": 0.3,
            "skills": 0.25,
            "experience": 0.2,
            "education": 0.15,
            "projects": 0.05,
            "certificates": 0.03,
            "publications": 0.02
        }
        
    def calculate_experience_score(self, resume_exp: str, jd_exp: str | List[str]) -> Tuple[float, float]:
        """Calculate experience matching score"""
        def extract_years(text: str) -> float:
            if not text:
                return 0.0
                
            # Look for patterns like "X years", "X+ years", "X-Y years"
            patterns = [
                r'(\d+)\+?\s*(?:years?|yrs?)',  # Matches "5 years", "5+ years"
                r'(\d+)[-–](\d+)\s*(?:years?|yrs?)',  # Matches "3-5 years"
                r'(\d+)\s*(?:years?|yrs?)\s+experience'  # Matches "5 years experience"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    if isinstance(matches[0], tuple):  # Range pattern matched
                        return (float(matches[0][0]) + float(matches[0][1])) / 2  # Average the range
                    return float(matches[0])
            return 0.0
        
        resume_years = extract_years(resume_exp)
        if not resume_years and isinstance(resume_exp, str) and "experience" in resume_exp.lower():
            # Fallback: count years from work experience dates
            resume_years = extract_years(resume_exp)
        
        # Handle JD experience which could be a string or list
        if isinstance(jd_exp, list):
            jd_exp_text = " ".join(jd_exp)
        else:
            jd_exp_text = jd_exp
            
        jd_years = extract_years(jd_exp_text)
        
        if jd_years == 0 or "no specific experience requirement" in jd_exp_text.lower():
            return 1.0, resume_years if resume_years > 0 else 0.0
        
        score = min(1.0, resume_years / jd_years)
        return score, resume_years
    
    def calculate_skill_match(self, resume_skills: List[str], jd_skills: List[str], resume_data: Dict[str, Any] = None) -> Tuple[float, List[str], List[str], List[Tuple[str, str, float]]]:
        """Enhanced skill matching with better semantic similarity"""
        if not jd_skills and resume_data:
            # If no JD skills provided, extract from resume summary and work experience
            jd_skills = []
            if isinstance(resume_data.get("basics"), dict) and resume_data["basics"].get("summary"):
                summary_skills = ResumeDataCleaner.extract_skills_from_text(resume_data["basics"]["summary"])
                jd_skills.extend(summary_skills)
            
            if isinstance(resume_data.get("work"), list):
                for job in resume_data["work"]:
                    if isinstance(job.get("highlights"), list):
                        for highlight in job["highlights"]:
                            highlight_skills = ResumeDataCleaner.extract_skills_from_text(str(highlight))
                            jd_skills.extend(highlight_skills)
            
            # Clean and deduplicate extracted skills
            jd_skills = list(set([s.strip() for s in jd_skills if s.strip()]))
            
        matched = []
        missing = []
        semantic_matches = []
        
        def normalize_skill(skill: str) -> str:
            """Normalize skill names for better matching"""
            skill = skill.lower().strip()
            replacements = {
                'javascript': 'js',
                'node.js': 'nodejs',
                'c++': 'cpp',
                'c#': 'csharp',
                'rest api': 'rest',
                'restful': 'rest',
                'rest apis': 'rest',
                'springboot': 'spring boot',
                'postgresql': 'postgres',
                'micro service': 'microservices',
                'micro-services': 'microservices',
                'cicd': 'ci/cd',
                'ci/cd': 'ci/cd',
                'ci-cd': 'ci/cd',
                'devops': 'ci/cd'
            }
            for old, new in replacements.items():
                if old in skill:
                    skill = skill.replace(old, new)
            return skill
        
        def skills_similar(skill1: str, skill2: str) -> float:
            """Calculate similarity between two skills"""
            s1, s2 = normalize_skill(skill1), normalize_skill(skill2)
            
            # Exact match
            if s1 == s2:
                return 1.0
            
            # Handle common variations
            rest_variations = ['rest', 'restful', 'rest api', 'rest apis', 'restful api', 'restful apis']
            if (s1 in rest_variations and s2 in rest_variations):
                return 1.0
            
            # Handle Spring variations
            if ('spring' in s1 and 'spring' in s2) or (s1 == 'spring' and 'spring boot' in s2):
                return 0.95
            
            # Handle CI/CD variations
            cicd_variations = ['ci/cd', 'cicd', 'ci-cd', 'devops']
            if (s1 in cicd_variations and s2 in cicd_variations):
                return 1.0
            
            # Handle database variations
            db_variations = {
                'postgres': ['postgresql', 'postgres db', 'postgresql db'],
                'mysql': ['mysql db', 'mysql database'],
                'mongodb': ['mongo', 'mongo db', 'mongodb database']
            }
            for db, variations in db_variations.items():
                if (s1 == db and s2 in variations) or (s2 == db and s1 in variations):
                    return 0.95
            
            # Substring match
            if s1 in s2 or s2 in s1:
                return 0.9
            
            # Use SequenceMatcher for fuzzy matching
            return SequenceMatcher(None, s1, s2).ratio()
        
        # First clean JD skills (might have parsing artifacts)
        cleaned_jd_skills = []
        for skill in jd_skills:
            skill = skill.strip()
            if skill and not skill.lower().endswith('requirements:'):
                cleaned_jd_skills.append(skill)
        jd_skills = cleaned_jd_skills
        
        # Find matches for each JD skill
        for jd_skill in jd_skills:
            best_match = None
            best_score = 0
            
            for resume_skill in resume_skills:
                similarity = skills_similar(jd_skill, resume_skill)
                if similarity > best_score:
                    best_score = similarity
                    best_match = resume_skill
            
            if best_score >= 0.85:  # High similarity threshold for direct match
                matched.append(jd_skill)
            elif best_score >= 0.6:  # Medium similarity for semantic match
                semantic_matches.append((best_match, jd_skill, best_score))
            else:
                missing.append(jd_skill)
        
        # Calculate final score
        total_jd_skills = len(jd_skills)
        if total_jd_skills == 0:
            return 0.0, [], [], []
        
        direct_matches = len(matched)
        semantic_match_score = sum(score for _, _, score in semantic_matches)
        
        final_score = (direct_matches + semantic_match_score) / total_jd_skills
        
        return min(1.0, final_score), matched, missing, semantic_matches
    
    def analyze_match(self, resume_data: Dict[str, Any], jd_data: Dict[str, Any], semantic_score: float) -> MatchResults:
        """Analyze match between resume and job description"""
        # Clean and extract resume data
        resume_data["basics"] = ResumeDataCleaner.clean_basics(resume_data.get("basics", {}))
        resume_data["work"] = ResumeDataCleaner.clean_work_experience(
            resume_data.get("work", []),
            json.dumps(resume_data, indent=2)  # Use full resume as text fallback
        )
        
        # Fix education parsing
        if isinstance(resume_data.get("education"), list):
            for edu in resume_data["education"]:
                if isinstance(edu, dict):
                    # Split area and institution if combined
                    area = edu.get("area", "")
                    if "  " in area:  # Double space indicates separation
                        parts = area.split("  ")
                        edu["area"] = parts[0].strip()
                        edu["institution"] = parts[1].strip()
                    elif "|" in area:  # Pipe separator
                        parts = area.split("|")
                        edu["area"] = parts[0].strip()
                        edu["institution"] = parts[1].strip()
                    elif "-" in area:  # Dash separator
                        parts = area.split("-")
                        edu["area"] = parts[0].strip()
                        edu["institution"] = parts[1].strip()
        
        resume_data["education"] = ResumeDataCleaner.clean_education(
            resume_data.get("education", []),
            json.dumps(resume_data, indent=2)
        )
        
        # Clean and deduplicate skills
        resume_skills = ResumeDataCleaner.extract_skills(resume_data)
        # Normalize REST-related skills
        normalized_skills = []
        seen_skills = set()
        for skill in resume_skills:
            skill_lower = skill.lower()
            if skill_lower in ['rest', 'restful', 'rest api', 'rest apis', 'restful api', 'restful apis']:
                if 'rest' not in seen_skills:
                    normalized_skills.append("REST APIs")
                    seen_skills.add('rest')
            elif skill_lower not in seen_skills:
                normalized_skills.append(skill)
                seen_skills.add(skill_lower)
        resume_data["skills"] = [{"name": skill} for skill in normalized_skills]
        
        # Extract JD requirements
        jd_skills = []
        jd_exp = ""
        jd_qualifications = []
        if isinstance(jd_data, dict):
            jd_skills = jd_data.get("requirements", {}).get("skills", [])
            if isinstance(jd_skills, str):
                jd_skills = [s.strip() for s in jd_skills.split(',') if s.strip()]
            jd_exp = jd_data.get("requirements", {}).get("experience", [])
            if isinstance(jd_exp, list):
                jd_exp = " ".join(jd_exp)
            jd_qualifications = jd_data.get("requirements", {}).get("qualifications", [])
            if isinstance(jd_qualifications, str):
                jd_qualifications = [jd_qualifications]
        
        # Calculate skill match
        skill_score, matched_skills, missing_skills, semantic_matches = self.calculate_skill_match(
            normalized_skills, jd_skills, resume_data
        )
        
        # Calculate experience match with improved date handling
        resume_exp = ""
        total_years = 0.0
        if isinstance(resume_data.get("work"), list):
            for job in resume_data["work"]:
                if job.get("startDate") and job.get("endDate"):
                    try:
                        # Handle different date formats
                        start_date = job["startDate"]
                        end_date = job["endDate"]
                        
                        # Convert month-year format to datetime
                        if len(start_date.split()) == 2:  # "Jun 2019" format
                            start = datetime.strptime(start_date, "%b %Y")
                        else:
                            start = datetime.strptime(start_date, "%Y-%m-%d")
                            
                        if end_date.lower() in ['present', 'current']:
                            end = datetime.now()
                        elif len(end_date.split()) == 2:  # "Sep 2023" format
                            end = datetime.strptime(end_date, "%b %Y")
                        else:
                            end = datetime.strptime(end_date, "%Y-%m-%d")
                            
                        # Calculate years with decimal precision
                        years = (end - start).days / 365.25
                        total_years += years
                    except Exception as e:
                        print(f"Error parsing dates: {e}")
                        continue
                        
            # Round to 1 decimal place for display
            resume_exp = f"{round(total_years, 1)} years of experience"
        
        exp_score, resume_years = self.calculate_experience_score(resume_exp, jd_exp)
        
        # Calculate education match
        edu_score = 0.0
        if isinstance(resume_data.get("education"), list) and len(resume_data["education"]) > 0:
            edu_score = 1.0  # Basic score for having education
            # Bonus if degree matches requirements
            for qual in jd_qualifications:
                qual_lower = qual.lower()
                if "bachelor" in qual_lower or "degree" in qual_lower:
                    for edu in resume_data["education"]:
                        if "bachelor" in edu.get("studyType", "").lower():
                            edu_score = 1.2  # Slight boost for matching degree
                            break
        
        # Calculate project match
        project_score = 0.0
        if isinstance(resume_data.get("projects"), list):
            project_score = min(1.0, len(resume_data["projects"]) / 3)  # Normalize to 0-1
        
        # Calculate certificate match
        cert_score = 0.0
        if isinstance(resume_data.get("certificates"), list):
            cert_score = min(1.0, len(resume_data["certificates"]) / 3)  # Normalize to 0-1
        
        # Calculate publication match
        pub_score = 0.0
        if isinstance(resume_data.get("publications"), list):
            pub_score = min(1.0, len(resume_data["publications"]) / 2)  # Normalize to 0-1
        
        # Calculate overall match percentage
        match_percentage = (
            semantic_score * self.weights["semantic"] +
            skill_score * self.weights["skills"] +
            exp_score * self.weights["experience"] +
            edu_score * self.weights["education"] +
            project_score * self.weights["projects"] +
            cert_score * self.weights["certificates"] +
            pub_score * self.weights["publications"]
        ) * 100
        
        # Create match details
        details = MatchDetails(
            semantic_score=semantic_score,
            skill_score=skill_score,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            semantically_matched_skills=semantic_matches,
            experience_score=exp_score,
            education_score=edu_score,
            calculated_resume_experience_years=total_years,  # Use the calculated total_years
            weights=self.weights,
            basics=resume_data.get("basics"),
            work=resume_data.get("work"),
            education=resume_data.get("education"),
            publications=resume_data.get("publications"),
            skills=resume_data.get("skills"),
            languages=resume_data.get("languages"),
            projects=resume_data.get("projects"),
            certificates=resume_data.get("certificates")
        )
        
        return MatchResults(
            match_percentage=min(100.0, max(0.0, match_percentage)),  # Ensure between 0-100
            details=details,
            parsed_resume=resume_data
        )
    
    def to_json(self, results: MatchResults) -> str:
        """Convert match results to JSON string with better formatting"""
        result_dict = asdict(results)
        
        # Add summary for better readability
        summary = {
            "match_summary": {
                "overall_match": f"{results.match_percentage:.1f}%",
                "semantic_match": f"{results.details.semantic_score:.2f}",
                "skills_matched": f"{len(results.details.matched_skills)}/{len(results.details.matched_skills) + len(results.details.missing_skills)}",
                "experience_match": f"{results.details.experience_score:.2f}",
                "candidate_experience": f"{results.details.calculated_resume_experience_years:.1f} years",
                "education_match": f"{results.details.education_score:.2f}"
            }
        }
        
        result_dict.update(summary)
        return json.dumps(result_dict, indent=2)

# Example usage:
if __name__ == "__main__":
    # Sample resume data (would normally come from parser)
    resume_data = {
        "basics": {
            "name": "RAVI SHARMA",
            "label": "Backend Developer | Bangalore, India",
            "email": "ravi.sharma@example.com",
            "phone": "+91-9876543210",
            "summary": "Backend developer with 4+ years of experience in building scalable microservices using Java, Spring Boot, and PostgreSQL.",
            "profiles": [{"network": "LinkedIn", "url": "linkedin.com/in/ravisharma"}]
        },
        "work": [
            {
                "company": "Infosys",
                "position": "Software Engineer",
                "startDate": "2019-06",
                "endDate": "2023-09",
                "highlights": [
                    "Developed RESTful services using Spring Boot and PostgreSQL.",
                    "Migrated legacy applications to microservices architecture.",
                    "Integrated CI/CD pipelines using Jenkins."
                ]
            }
        ],
        "education": [
            {
                "institution": "IIT Delhi",
                "area": "Computer Science",
                "studyType": "B.Tech",
                "startDate": "2015",
                "endDate": "2019"
            }
        ],
        "skills": [
            {"name": "Java"},
            {"name": "Spring Boot"},
            {"name": "Microservices"},
            {"name": "REST APIs"},
            {"name": "PostgreSQL"},
            {"name": "Git"}
        ]
    }
    
    # Sample JD data
    jd_data = {
        "title": "Java Backend Developer",
        "requirements": {
            "skills": ["Java", "Spring Boot", "Microservices", "REST APIs", "SQL", "Git"],
            "experience": "3+ years of experience with backend systems",
            "education": "Bachelor's degree in Computer Science or related field"
        }
    }
    
    analyzer = MatchAnalyzer()
    
    # Normally you would get semantic_score from embedding comparison
    semantic_score = 0.92  # Example value
    
    results = analyzer.analyze_match(resume_data, jd_data, semantic_score)
    print(analyzer.to_json(results))