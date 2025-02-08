import pandas as pd
import time
from serpapi import GoogleSearch

# Set your SerpAPI key here
SERPAPI_KEY = "156ee562f5f53de280e8d5442bae4f1d5107023ecdf2c0a0e2f8bc83ede64e65"
def get_salary_from_web(job_title):
    """Fetch the average salary of a job title in India (LPA) using SerpAPI."""
    query = f"Average salary of {job_title} in India (LPA)"
    params = {
        "q": query,
        "location": "India",
        "hl": "en",
        "gl": "in",
        "api_key": SERPAPI_KEY
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    return extract_salary(results)

def extract_salary(results):
    """Extract salary information from SerpAPI search results."""
    import re
    if "organic_results" in results:
        for result in results["organic_results"]:
            text = result.get("snippet", "")
            matches = re.findall(r'\d+\.?\d*\s*LPA', text)
            if matches:
                return matches[0]
    return None

# Load the dataset
file_path = r"C:\Users\gouri\OneDrive\Desktop\PROJECTS\TrendJobs\cleaned_it_jobs_data_filled_skills.csv"
df = pd.read_csv(file_path)

# Process missing salary values
missing_titles = df[df['salary'].isnull()]['title'].unique()

for title in missing_titles:
    estimated_salary = get_salary_from_web(title)
    if estimated_salary:
        df.loc[(df['title'] == title) & (df['salary'].isnull()), 'salary'] = estimated_salary
    time.sleep(2)  # Prevent excessive requests

# Save the updated dataset
df.to_csv("updated_salaries.csv", index=False)
print("Missing salaries filled and saved to updated_salaries.csv")
