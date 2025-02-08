import requests
import json
from chromadb import Client
from chromadb.config import Settings

# Your SearchAPI key for Google Jobs
API_KEY = "SearchAPI"
URL = "URL to SearchAPi"  # API endpoint for job listings

# List of IT job titles you want to query
job_titles = [
    "Data Scientist", "Software Engineer", "Cybersecurity Specialist", "Artificial Intelligence Engineer",
    "Machine Learning Engineer", "Full Stack Developer", "Cloud Computing Professional", "DevOps Engineer",
    "UX/UI Designer", "Healthcare Data Analyst", "Web Developer", "Database Administrator", 
    "Network Architect", "IT Project Manager", "Business Analyst", "Quality Assurance Tester", 
    "Technical Writer", "IT Consultant", "Network Engineer", "Systems Administrator", "Help Desk Technician",
    "Computer Systems Analyst", "Information Security Manager", "IT Manager", "Network Security Engineer", 
    "Cloud Architect", "Data Analyst", "Business Intelligence Developer", "IT Auditor", "Cybersecurity Consultant", 
    "IT Trainer", "Technical Support Specialist", "Software Developer", "Data Engineer", "AI Researcher",
    "Computer Vision Engineer", "Natural Language Processing Engineer", "Robotics Engineer", 
    "Human-Computer Interaction Designer", "User Experience Researcher", "IT Compliance Officer", 
    "IT Risk Manager", "IT Governance Manager", "IT Service Manager", "IT Asset Manager"
]

# Define function to collect job data for a specific title
def collect_job_data(query):
    params = {
        "engine": "google_jobs",  # Specify the search engine
        "q": query,               # Use the query for job title
        "api_key": API_KEY        # Use the provided API key
    }
    response = requests.get(URL, params=params)
    if response.status_code == 200:
        return response.json()  # Return JSON data
    else:
        print(f"Failed to fetch data for {query}: {response.status_code}")
        return None

# Save data in JSON format
def save_data(data, filename="jobs_data.json"):
    with open(filename, "w") as file:
        json.dump(data, file, indent=4)  # Save data as JSON
        print(f"Data saved as {filename}")

# Initialize ChromaDB client
def initialize_chromadb():
    client = Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chromadb"))
    return client

# Save job data to ChromaDB
def save_to_chromadb(data):
    client = initialize_chromadb()
    collection = client.create_collection("job_listings")  # Create a collection for job listings

    if "jobs" in data:
        for job in data["jobs"]:
            description = job.get("description", "No description available")
            title = job.get("title", "No title")
            company = job.get("company", "No company")
            location = job.get("location", "No location")
            salary = job.get("salary", "Not provided")

            collection.add(
                documents=[description],
                metadatas=[{
                    "title": title,
                    "company": company,
                    "location": location,
                    "salary": salary
                }],
                ids=[job.get("id", "no_id")]
            )
        print("Job data successfully saved to ChromaDB")
    else:
        print("No jobs found in the API response")

# Main execution: Collect job data for each title and store in ChromaDB
if __name__ == "__main__":
    all_jobs_data = {"jobs": []}  # Initialize a dictionary to store all jobs
    for title in job_titles:
        print(f"Fetching data for: {title}")
        job_data = collect_job_data(title)
        if job_data:
            all_jobs_data["jobs"].extend(job_data.get("jobs", []))  # Add jobs to the collection

    if all_jobs_data["jobs"]:
        save_data(all_jobs_data, "all_it_jobs_data.json")  # Save as JSON file
        save_to_chromadb(all_jobs_data)  # Save to ChromaDB
