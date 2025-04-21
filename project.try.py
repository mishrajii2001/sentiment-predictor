import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import os
import csv

# Sample dataset
data = [
    ("Terrible experience, would not recommend.", 1),
    ("Pretty decent quality, fits well.", 4),
    ("Not as expected. Poor stitching.", 2),
    ("This product is amazing!", 5),
    ("Absolutely wonderful, exceeded expectations!", 5),
    ("Broke after a week. Not worth it.", 2),
    ("Nice design and comfortable.", 4),
    ("Didn't fit properly. Cheap material.", 2),
    ("Perfect! Would buy again.", 5),
    ("Works well, no complaints.", 4),
    ("The material feels cheap and it ripped quickly.", 2),
    ("High quality and fast delivery!", 5),
    ("Looks different than the pictures.", 2),
    ("Fits perfectly and looks great.", 4),
    ("Color faded after one wash.", 2),
    ("Exactly as described. Loved it!", 5),
    ("Doesn't match the description.", 2)
]

# Train model
reviews, labels = zip(*data)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)
model = MultinomialNB()
model.fit(X, labels)

# File to store reviews
REVIEW_FILE = "review_log.csv"

# Load existing reviews
def load_reviews():
    if os.path.exists(REVIEW_FILE):
        return pd.read_csv(REVIEW_FILE)
    else:
        return pd.DataFrame(columns=["Star", "Review", "Sentiment"])

# Save a new review
def save_review(star, review, sentiment):
    with open(REVIEW_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([star, review, sentiment])

# Streamlit UI
st.set_page_config(page_title="Sentiment Predictor", layout="centered")
st.title("🔷 Sentiment Predictor App")
st.markdown("Predict whether a product review is **Positive** or **Negative** using a trained ML model.")

# User Input
star = st.number_input("⭐ Star Rating (1-5)", min_value=1, max_value=5, step=1)
review = st.text_input("💬 Enter Your Review")

if st.button("▶ Predict"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:
        X_input = vectorizer.transform([review])
        predicted_rating = model.predict(X_input)[0]
        sentiment = "Positive" if predicted_rating >= 3 else "Negative"

        st.success(f"🧠 Predicted Sentiment: **{sentiment}**")
        save_review(star, review, sentiment)

# Show Review Log
st.markdown("### 📋 Review Log")
review_data = load_reviews()
st.dataframe(review_data[::-1])  # reverse to show latest first




#python -m streamlit run "C:/Users/VICKY MISHRA/Desktop/project.try.py"