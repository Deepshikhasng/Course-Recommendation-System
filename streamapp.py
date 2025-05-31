import streamlit as st
import pandas as pd
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
import re

# ----------------------------
# TextCleaner class (must match the one used in your pipeline)
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(self._clean_text)

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
# ----------------------------

st.set_page_config(page_title="Course Recommender", layout="wide")

@st.cache_resource
def load_data():
    df = pd.read_csv("cleaned_courses.csv")
    with open("course_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)
    return df, pipeline

df, course_pipeline = load_data()

# ----------------------------
def recommend_courses(query, top_n=10, difficulty=None, min_rating=0.0):
    cleaned_query = course_pipeline.named_steps["cleaner"].transform(pd.Series(query))
    query_vector = course_pipeline.named_steps["tfidf"].transform(cleaned_query)

    course_texts = course_pipeline.named_steps["cleaner"].transform(df["course_title"])
    course_vectors = course_pipeline.named_steps["tfidf"].transform(course_texts)

    sim_scores = cosine_similarity(query_vector, course_vectors).flatten()
    df["similarity"] = sim_scores
    filtered_df = df.copy()

    if difficulty != "All":
        filtered_df = filtered_df[filtered_df["course_difficulty"] == difficulty]

    filtered_df = filtered_df[filtered_df["course_rating"] >= min_rating]
    top_results = filtered_df.sort_values(by="similarity", ascending=False).head(top_n)

    return top_results[[ 
        "course_title", 
        "course_organization", 
        "course_rating", 
        "course_difficulty", 
        "course_students_enrolled"
    ]]
# ----------------------------

# ----------------------------
# Streamlit UI
st.markdown("<h1 style='color:#1f77b4;'>🎓 Course Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("Get personalized course suggestions based on your skills or career goals.")

user_input = st.text_input("Enter your skills or goals (e.g., Python, AI, Data Science):")

col1, col2 = st.columns(2)
difficulty = col1.selectbox("Filter by Difficulty", options=["All", "Beginner", "Intermediate", "Advanced"])
min_rating = col2.slider("Minimum Rating", 0.0, 5.0, 3.5, step=0.1)

search_button = st.button("🔍 Search")

if search_button and user_input:
    with st.spinner("Finding the best courses for you..."):
        try:
            results = recommend_courses(user_input, difficulty=difficulty, min_rating=min_rating)
            if not results.empty:
                st.success("Top Recommended Courses:")
                for idx, row in results.iterrows():
                    st.markdown(f"""
                    <div style="
                        background-color: #ffffff;
                        color: #000000;
                        padding: 15px;
                        margin-bottom: 10px;
                        border-radius: 10px;
                        border: 1px solid #ddd;">
                        <h4 style="margin-bottom:5px;">{row['course_title']}</h4>
                        <p>
                            <b>Organization:</b> {row['course_organization']} |
                            <b>Rating:</b> {row['course_rating']} ⭐ |
                            <b>Difficulty:</b> {row['course_difficulty']} |
                            <b>Enrolled:</b> {row['course_students_enrolled']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No matching courses found. Try different filters or keywords.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
# ----------------------------
