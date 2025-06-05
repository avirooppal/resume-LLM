import re
from typing import Dict, List, Any
import PyPDF2
import docx
import json
from config.settings import settings
import os

class ResumePreprocessor:
    def __init__(self):
        """Initialize the resume preprocessor"""
        pass
        
    def validate_input(self, text: str) -> bool:
        """
        Validate resume text meets minimum requirements.
        
        Args:
            text (str): Resume text to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
            
        # Check if text has minimum required sections
        required_sections = ['EXPERIENCE', 'EDUCATION', 'SKILLS']
        text_upper = text.upper()
        has_required_sections = all(section in text_upper for section in required_sections)
        
        # Check minimum length
        has_min_length = len(text.strip()) >= 100  # Minimum 100 characters
        
        return has_required_sections and has_min_length
        
    def clean_text(self, input_data: str) -> str:
        """
        Clean and extract text from a resume file or text.
        
        Args:
            input_data (str): Path to the resume file (PDF or DOCX) or text content
            
        Returns:
            str: Cleaned text content
        """
        # If input_data is a file path
        if os.path.isfile(input_data):
            # Extract text based on file type
            if input_data.lower().endswith('.pdf'):
                text = self._extract_from_pdf(input_data)
            elif input_data.lower().endswith('.docx'):
                text = self._extract_from_docx(input_data)
            else:
                raise ValueError(f"Unsupported file type: {input_data}")
        else:
            # If input_data is the text content itself
            text = input_data
            
        # Clean the text
        text = self._clean_text_content(text)
        return text
        
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
        
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
    def _clean_text_content(self, text: str) -> str:
        """Clean the extracted text"""
        # Normalize line breaks
        text = re.sub(r'\r\n|\r|\n', '\n', text)
        # Remove extra whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove special characters but keep newlines and important punctuation
        text = re.sub(r'[^\w\s\n.,;:!?@#$%&*()\-+=<>/]', '', text)
        # Normalize newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
        
    def extract_sections(self, text: str) -> Dict[str, Any]:
        """
        Extract sections from resume text.
        
        Args:
            text (str): Cleaned resume text
            
        Returns:
            Dict[str, Any]: Extracted sections
        """
        sections = {
            "basics": self._extract_basics(text),
            "work": self._extract_work_experience(text),
            "education": self._extract_education(text),
            "skills": self._extract_skills(text),
            "publications": [],
            "languages": [],
            "projects": [],
            "certificates": []
        }
        return sections
        
    def _extract_basics(self, text: str) -> Dict[str, Any]:
        """Extract basic information"""
        basics = {
            "name": "",
            "email": "",
            "phone": "",
            "summary": "",
            "location": {"city": "", "region": ""},
            "profiles": []
        }
        
        # Split text into lines and clean
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Extract name (first line)
        if lines:
            basics["name"] = lines[0].strip()
            
        # Extract email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            basics["email"] = email_match.group()
            
        # Extract phone
        phone_match = re.search(r'(?:\+?91[-.\s]?)?[6-9]\d{9}', text)
        if phone_match:
            basics["phone"] = phone_match.group()
            
        # Extract location
        location_match = re.search(r'\b(Bangalore|Mumbai|Delhi|Hyderabad|Chennai)\b', text, re.IGNORECASE)
        if location_match:
            basics["location"]["city"] = location_match.group(1)
            
        # Extract LinkedIn profile
        linkedin_match = re.search(r'(https?://(?:www\.)?linkedin\.com/[^\s]+)', text)
        if linkedin_match:
            basics["profiles"].append({
                "network": "LinkedIn",
                "url": linkedin_match.group(1)
            })
            
        # Extract summary (only the first paragraph before SKILLS)
        summary_lines = []
        found_summary = False
        for line in lines:
            if line.upper() == 'SKILLS':
                break
            if found_summary:
                summary_lines.append(line)
            elif line.upper() == 'SUMMARY':
                found_summary = True
                
        if summary_lines:
            basics["summary"] = ' '.join(summary_lines).strip()
            
        return basics
        
    def _extract_work_experience(self, text: str) -> List[Dict[str, Any]]:
        """Extract work experience"""
        work_experience = []
        seen_jobs = set()  # Track unique jobs
        
        # Split text into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Find experience section
        exp_start = -1
        for i, line in enumerate(lines):
            if line.upper() == 'EXPERIENCE':
                exp_start = i + 1
                break
                
        if exp_start != -1:
            current_job = None
            for line in lines[exp_start:]:
                if line.upper() in ['EDUCATION', 'SKILLS', 'PROJECTS']:
                    if current_job:
                        job_key = f"{current_job['company']}|{current_job['position']}|{current_job['startDate']}|{current_job['endDate']}"
                        if job_key not in seen_jobs:
                            seen_jobs.add(job_key)
                            work_experience.append(current_job)
                    break
                    
                # Look for job title and company
                job_match = re.match(r'^([A-Z][A-Za-z\s]+(?:Engineer|Developer|Analyst|Manager|Lead|Architect))\s*(?:[-|]\s*|\s+)([A-Za-z\s]+)', line)
                if job_match:
                    if current_job:
                        job_key = f"{current_job['company']}|{current_job['position']}|{current_job['startDate']}|{current_job['endDate']}"
                        if job_key not in seen_jobs:
                            seen_jobs.add(job_key)
                            work_experience.append(current_job)
                    current_job = {
                        "company": job_match.group(2).strip(),
                        "position": job_match.group(1).strip(),
                        "startDate": "",
                        "endDate": "",
                        "highlights": []
                    }
                    # Look for dates in the same line
                    date_match = re.search(r'(\w{3}\s+\d{4})\s*[-–]\s*(\w{3}\s+\d{4}|\w+\s+\d{4}|Present|Current)', line)
                    if date_match:
                        current_job["startDate"] = date_match.group(1)
                        current_job["endDate"] = date_match.group(2)
                    continue
                    
                # Look for dates if not found in job title line
                if current_job and not current_job["startDate"]:
                    date_match = re.search(r'(\w{3}\s+\d{4})\s*[-–]\s*(\w{3}\s+\d{4}|\w+\s+\d{4}|Present|Current)', line)
                    if date_match:
                        current_job["startDate"] = date_match.group(1)
                        current_job["endDate"] = date_match.group(2)
                        continue
                        
                # Look for highlights
                if current_job and line.strip().startswith(('-', '•', '*')):
                    highlight = line.strip().lstrip('-•* ').strip()
                    if highlight and highlight not in current_job["highlights"]:
                        current_job["highlights"].append(highlight)
                    
            # Add the last job if not duplicate
            if current_job:
                job_key = f"{current_job['company']}|{current_job['position']}|{current_job['startDate']}|{current_job['endDate']}"
                if job_key not in seen_jobs:
                    seen_jobs.add(job_key)
                    work_experience.append(current_job)
                
        return work_experience
        
    def _extract_education(self, text: str) -> List[Dict[str, Any]]:
        """Extract education information"""
        education = []
        
        # Split text into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Find education section
        edu_start = -1
        for i, line in enumerate(lines):
            if line.upper() == 'EDUCATION':
                edu_start = i + 1
                break
                
        if edu_start != -1:
            current_edu = None
            for line in lines[edu_start:]:
                if line.upper() in ['SKILLS', 'PROJECTS', 'CERTIFICATIONS']:
                    if current_edu:
                        education.append(current_edu)
                    break
                    
                # Look for degree
                degree_match = re.match(r'^(B\.?Tech|B\.?E\.?|Bachelor|M\.?Tech|M\.?E\.?|Master|PhD|Diploma)\s+in\s+([A-Za-z\s]+)', line, re.IGNORECASE)
                if degree_match:
                    if current_edu:
                        education.append(current_edu)
                    current_edu = {
                        "institution": "",
                        "area": degree_match.group(2).strip(),
                        "studyType": degree_match.group(1).strip(),
                        "startDate": "",
                        "endDate": ""
                    }
                    continue
                    
                # Look for institution and years
                if current_edu:
                    # Check if line contains years
                    years = re.findall(r'\d{4}', line)
                    if len(years) >= 2:
                        current_edu["startDate"] = f"{years[0]}-01-01"
                        current_edu["endDate"] = f"{years[1]}-12-31"
                        # Remove years from line to get institution
                        line = re.sub(r'\d{4}\s*[-–]\s*\d{4}', '', line).strip()
                    
                    # If institution is empty, use the line
                    if not current_edu["institution"] and line:
                        # First try to extract institution using common patterns
                        institution_patterns = [
                            r'(?:from|at)\s+([A-Za-z\s]+(?:University|Institute|College|IIT|IIIT|NIT))',
                            r'([A-Za-z\s]+(?:University|Institute|College|IIT|IIIT|NIT))',
                            r'([A-Za-z\s]+(?:University|Institute|College|IIT|IIIT|NIT))\s*[-–|]\s*[A-Za-z\s]+'
                        ]
                        
                        for pattern in institution_patterns:
                            institution_match = re.search(pattern, line, re.IGNORECASE)
                            if institution_match:
                                current_edu["institution"] = institution_match.group(1).strip()
                                # Remove institution from line to get area if needed
                                line = line.replace(institution_match.group(1), '').strip()
                                if not current_edu["area"] and line:
                                    current_edu["area"] = line.strip()
                                break
                        
                        # If no institution found, try splitting by common separators
                        if not current_edu["institution"]:
                            parts = re.split(r'\s*\|\s*|\s*[-–]\s*', line)
                            if len(parts) > 1:
                                current_edu["institution"] = parts[0].strip()
                                if not current_edu["area"]:
                                    current_edu["area"] = parts[1].strip()
                            else:
                                # If still no institution, use the whole line
                                current_edu["institution"] = line.strip()
                        
            # Add the last education entry
            if current_edu:
                education.append(current_edu)
                
        return education
        
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills"""
        skills = set()  # Use set to prevent duplicates
        
        # Split text into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Find skills section
        skills_start = -1
        for i, line in enumerate(lines):
            if line.upper() == 'SKILLS':
                skills_start = i + 1
                break
                
        if skills_start != -1:
            for line in lines[skills_start:]:
                if line.upper() in ['EXPERIENCE', 'EDUCATION', 'PROJECTS']:
                    break
                    
                # Extract skills from bullet points
                if line.strip().startswith(('-', '•', '*')):
                    # Split by commas and clean
                    point_skills = [s.strip() for s in line.strip().lstrip('-•* ').split(',')]
                    for skill in point_skills:
                        if skill:
                            # Normalize skill names
                            skill = skill.strip()
                            # Normalize REST-related skills
                            if any(rest in skill.lower() for rest in ['restful', 'rest api', 'rest apis']):
                                skill = 'REST APIs'
                            # Normalize other common variations
                            skill = skill.replace('SpringBoot', 'Spring Boot')
                            skill = skill.replace('PostgreSQL', 'Postgres')
                            skill = skill.replace('Micro service', 'Microservices')
                            skills.add(skill)
                    
        # Clean and deduplicate skills
        cleaned_skills = []
        for skill in skills:
            if skill and not re.match(r'\d{4}', skill) and not skill.endswith('.'):
                cleaned_skills.append({"name": skill})
                
        return cleaned_skills

class JDPreprocessor:
    def __init__(self):
        """Initialize the job description preprocessor"""
        pass
        
    def clean_text(self, input_data: str) -> str:
        """
        Clean and extract text from a job description file or text.
        
        Args:
            input_data (str): Path to the job description file or text content
            
        Returns:
            str: Cleaned text content
        """
        # If input_data is a file path
        if os.path.isfile(input_data):
            with open(input_data, 'r', encoding='utf-8') as file:
                text = file.read()
        else:
            # If input_data is the text content itself
            text = input_data
            
        # Clean the text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\n.,;:!?@#$%&*()\-+=<>/]', '', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
        
    def extract_requirements(self, text: str) -> Dict[str, Any]:
        """
        Extract requirements from job description text.
        
        Args:
            text (str): Cleaned job description text
            
        Returns:
            Dict[str, Any]: Extracted requirements
        """
        requirements = {
            "skills": [],
            "experience": [],
            "qualifications": []
        }
        
        # Split text into lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Extract skills
        skills_start = -1
        for i, line in enumerate(lines):
            if re.match(r'^Skills:?\s*$', line, re.IGNORECASE):
                skills_start = i + 1
                break
                
        if skills_start != -1:
            for line in lines[skills_start:]:
                if re.match(r'^(Experience|Qualifications|Requirements):?\s*$', line, re.IGNORECASE):
                    break
                if line.strip().startswith(('-', '•', '*')):
                    skills = [s.strip() for s in line.strip().lstrip('-•* ').split(',')]
                    requirements["skills"].extend([s for s in skills if s])
                    
        # Extract experience
        exp_start = -1
        for i, line in enumerate(lines):
            if re.match(r'^Experience:?\s*$', line, re.IGNORECASE):
                exp_start = i + 1
                break
                
        if exp_start != -1:
            for line in lines[exp_start:]:
                if re.match(r'^(Qualifications|Requirements):?\s*$', line, re.IGNORECASE):
                    break
                requirements["experience"].append(line.strip())
                
        # Extract qualifications
        qual_start = -1
        for i, line in enumerate(lines):
            if re.match(r'^Qualifications:?\s*$', line, re.IGNORECASE):
                qual_start = i + 1
                break
                
        if qual_start != -1:
            for line in lines[qual_start:]:
                if re.match(r'^Requirements:?\s*$', line, re.IGNORECASE):
                    break
                requirements["qualifications"].append(line.strip())
                    
        return requirements
    
    def validate_jd(self, text: str) -> bool:
        """Validate JD meets minimum requirements"""
        if not text or not isinstance(text, str):
            return False
            
        clean_text = self.clean_text(text)
        return len(clean_text) >= 50  # Minimum 50 characters for a valid JD