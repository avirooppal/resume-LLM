import pandas as pd
from typing import Dict, List, Set
import os

class DataLoader:
    def __init__(self):
        self.data_dir = "data"
        self.skills_df = None
        self.colleges_df = None
        self.companies_df = None
        self.load_data()

    def load_data(self):
        """Load all CSV files into memory"""
        try:
            # Load skills data (single column)
            self.skills_df = pd.read_csv(os.path.join(self.data_dir, "skills.csv"), header=None, names=['skill'])
            
            # Load colleges data (single column)
            self.colleges_df = pd.read_csv(os.path.join(self.data_dir, "colleges.csv"), header=None, names=['name'])
            
            # Load companies data (single column)
            self.companies_df = pd.read_csv(os.path.join(self.data_dir, "companies.csv"), header=None, names=['name'])
            
            print("Successfully loaded all data files")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def get_all_skills(self) -> Set[str]:
        """Get all unique skills"""
        if self.skills_df is None:
            return set()
        return set(self.skills_df['skill'].str.lower())

    def get_all_colleges(self) -> Set[str]:
        """Get all unique colleges"""
        if self.colleges_df is None:
            return set()
        return set(self.colleges_df['name'].str.lower())

    def get_all_companies(self) -> Set[str]:
        """Get all unique companies"""
        if self.companies_df is None:
            return set()
        return set(self.companies_df['name'].str.lower())

    def is_valid_skill(self, skill: str) -> bool:
        """Check if a skill exists in our database"""
        if self.skills_df is None:
            return False
        return skill.lower() in self.get_all_skills()

    def is_valid_college(self, college: str) -> bool:
        """Check if a college exists in our database"""
        if self.colleges_df is None:
            return False
        return college.lower() in self.get_all_colleges()

    def is_valid_company(self, company: str) -> bool:
        """Check if a company exists in our database"""
        if self.companies_df is None:
            return False
        return company.lower() in self.get_all_companies()

    def get_similar_skills(self, skill: str, threshold: float = 0.8) -> List[str]:
        """Find similar skills using fuzzy matching"""
        from difflib import SequenceMatcher
        
        if self.skills_df is None:
            return []
        
        similar_skills = []
        skill = skill.lower()
        
        for db_skill in self.get_all_skills():
            similarity = SequenceMatcher(None, skill, db_skill).ratio()
            if similarity >= threshold:
                similar_skills.append(db_skill)
        
        return similar_skills

# Initialize the data loader
data_loader = DataLoader() 