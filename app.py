import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def main():
    st.title("IMDb Movie Review Sentiment Analysis")
    review_text = st.text_area("Enter a movie review:")

    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')

        if st.button("Predict"):
            if review_text:
                review_processed = preprocess_text(review_text)
                review_vectorized = vectorizer.transform([review_processed])
                prediction = model.predict(review_vectorized)[0]
                if prediction == 1:
                    st.success("Positive Sentiment")
                else:
                    st.error("Negative Sentiment")
            else:
                st.warning("Please enter a review.")
    except FileNotFoundError:
        st.error("Model or Vectorizer files not found. Please train the model first.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()