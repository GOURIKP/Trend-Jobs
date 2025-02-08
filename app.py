import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import pickle
import matplotlib.pyplot as plt
import plotly.express as px

from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer

# ---- CONFIG ---- #
os.environ["GROQ_API_KEY"] = "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"

# ---- Load & Preprocess Data ---- #
DATA_PATH = r"C:\Users\gouri\OneDrive\Desktop\PROJECTS\TrendJobs\processed_jobs_data.csv"
df = pd.read_csv(DATA_PATH)

# Load FAISS Index
with open("faiss_index.pkl", "rb") as f:
    faiss_index = pickle.load(f)

# Load Sentence Transformer Model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---- Retrieve Jobs ---- #
def retrieve_jobs(query, top_n=10):
    """Retrieve top job roles based on query and adjust titles if necessary"""
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    _, indices = faiss_index.search(query_embedding, top_n * 2)  # Get extra jobs to filter

    results = df.iloc[indices[0]]

    # Define senior-related keywords
    senior_keywords = ["Senior", "Sr", "Lead", "Manager", "Director"]
    
    # Filter out roles explicitly marked as senior
    non_senior_jobs = results[~results['title'].str.contains('|'.join(senior_keywords), case=False, na=False)]

    if not non_senior_jobs.empty:
        return non_senior_jobs.head(top_n)
    
    # If no junior roles exist, modify senior job titles
    results["title"] = results["title"].replace(
        {kw: "" for kw in senior_keywords}, regex=True
    ).str.strip()  # Remove seniority keywords from the title
    
    return results.head(top_n)


# ---- Generate Natural Response using LLaMA 2 (Groq) ---- #
llm = ChatGroq(model_name="llama3-8b-8192")

job_prompt = PromptTemplate(
    input_variables=["skills", "jobs"],
    template="Based on the skills: {skills}, here are some great job roles: {jobs}. These roles align well with market trends and salaries."
)

job_chain = LLMChain(llm=llm, prompt=job_prompt)

def generate_response(user_query):
    jobs_df = retrieve_jobs(user_query, top_n=10)
    job_list = ", ".join(jobs_df['title'].tolist())
    response = job_chain.run(skills=user_query, jobs=job_list)
    return response, jobs_df

# ---- Visualization Functions ---- #
def plot_market_trends(jobs_df):
    fig = px.bar(
        jobs_df, 
        x="title", 
        y="salary", 
        color="salary",
        text="salary",
        title="Market Trend: Job Salary Comparison",
        labels={"salary": "Avg Salary ($)", "title": "Job Roles"},
        color_continuous_scale="blues"
    )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig)


def plot_salary_distribution(jobs_df):
    fig = px.line(
        jobs_df, 
        x="title", 
        y="salary", 
        title="Salary Trends Across Job Roles",
        labels={"salary": "Salary ($)", "title": "Job Roles"},
        markers=True  # Add markers to highlight points
    )
    fig.update_layout(xaxis_tickangle=-45, height=500)
    st.plotly_chart(fig)



def plot_skills_distribution(jobs_df):
    skills = " ".join(jobs_df['skills'].dropna().tolist()).split(", ")
    skills_series = pd.Series(skills).value_counts().nlargest(10)  # Show top 10 skills only
    
    fig = px.pie(
        names=skills_series.index, 
        values=skills_series.values, 
        title="Top Skills Required",
        hole=0.4,  # Donut-style chart
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    st.plotly_chart(fig)


# ---- Streamlit UI ---- #
st.title("💼 TrendJobs")
st.write("Enter your skills and job interests to find the best job opportunities based on market trends!")

user_query = st.text_input("Enter skills & job interests (e.g., Python, Data Science, ML):")

if st.button("Find Jobs") and user_query:
    response, jobs_df = generate_response(user_query)
    
    st.subheader("🔍 AI Recommendation:")
    st.write(response)
    
    st.subheader("📊 Job Market Analysis:")
    st.dataframe(jobs_df[['title', 'salary', 'skills']])
    
    st.subheader("📈 Market Trend Analysis")
    plot_market_trends(jobs_df)
    plot_salary_distribution(jobs_df)
    plot_skills_distribution(jobs_df)
