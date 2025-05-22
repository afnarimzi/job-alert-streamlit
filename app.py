import streamlit as st
import pandas as pd
from scraper import scrape_karkidi_jobs
from clustering import classify_new_jobs

st.title("🔍 Job Alert System - Karkidi.com")

user_interest = st.text_input("Enter your skills (comma separated):", "python, machine learning")

if st.button("Scrape and Match Jobs"):
    st.write("Scraping jobs...")
    jobs_df = scrape_karkidi_jobs(pages=2)
    jobs_df = classify_new_jobs(jobs_df)

    # Fake skill-to-cluster mapping based on previous runs
    interest_keywords = user_interest.lower().split(',')
    matching_clusters = jobs_df[jobs_df['Skills'].str.contains('|'.join(interest_keywords))]['Cluster'].unique()

    matched_jobs = jobs_df[jobs_df['Cluster'].isin(matching_clusters)]

    if not matched_jobs.empty:
        st.success(f"Found {len(matched_jobs)} matching jobs.")
        st.dataframe(matched_jobs[['Title', 'Company', 'Skills', 'Cluster']])
    else:
        st.warning("No matching jobs found.")
