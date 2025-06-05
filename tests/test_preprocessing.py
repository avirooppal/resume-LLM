import pytest
from modules.preprocessing import ResumePreprocessor, JDPreprocessor

def test_resume_clean_text():
    preprocessor = ResumePreprocessor()
    dirty_text = "  John   Doe  \n\n  <html>Skills: Python</html>  "
    clean_text = preprocessor.clean_text(dirty_text)
    assert "  " not in clean_text
    assert "<html>" not in clean_text
    assert clean_text == "John Doe Skills: Python"

def test_resume_validate_input():
    preprocessor = ResumePreprocessor()
    assert preprocessor.validate_input("Valid resume text with enough content") is True
    assert preprocessor.validate_input("Short") is False
    assert preprocessor.validate_input("") is False
    assert preprocessor.validate_input("   ") is False
    assert preprocessor.validate_input("12345") is False  # No letters

def test_jd_clean_text():
    jd_processor = JDPreprocessor()
    dirty_jd = "  Python   Developer  \n\n  <div>Requirements: 5 years</div>  "
    clean_jd = jd_processor.clean_jd(dirty_jd)
    assert "  " not in clean_jd
    assert "<div>" not in clean_jd
    assert clean_jd == "Python Developer Requirements: 5 years"

def test_jd_extract_requirements():
    jd_processor = JDPreprocessor()
    jd_text = """
    We require a Python developer. Must have 5 years experience.
    Skills required: Python, Django. Education: CS degree required.
    """
    requirements = jd_processor.extract_requirements(jd_text)
    assert len(requirements["skills"]) > 0
    assert len(requirements["qualifications"]) > 0
    assert len(requirements["experience"]) > 0