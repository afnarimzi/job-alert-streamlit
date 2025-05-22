import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib

def train_model(df, n_clusters=5):
    df['Skills'] = df['Skills'].fillna('').str.lower()
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Skills'])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X)

    joblib.dump(kmeans, 'model/kmeans_model.pkl')
    joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
    df.to_csv('data/jobs.csv', index=False)

    return df

def classify_new_jobs(df):
    kmeans = joblib.load('model/kmeans_model.pkl')
    vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

    df['Skills'] = df['Skills'].fillna('').str.lower()
    X = vectorizer.transform(df['Skills'])
    df['Cluster'] = kmeans.predict(X)

    return df
