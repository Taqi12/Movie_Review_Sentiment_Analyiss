# IMDb Movie Review Sentiment Analysis

This project builds a machine learning model to analyze the sentiment of movie reviews. It uses a Logistic Regression classifier trained on the IMDb movie reviews dataset.

## Usage

1.  Enter a movie review in the text area.
2.  Click the "Predict" button to check the review's sentiment.
3.  The app will display whether the sentiment is "Positive Sentiment" or "Negative Sentiment."

## Dataset

* The model is trained on the `IMDB Dataset.csv` dataset here is the dataset link https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews.
* Ensure the dataset has columns for the movie review text ("review") and sentiment labels ("sentiment").

## Model

* Logistic Regression is used for classification.
* TF-IDF vectorization is used for text representation.

## Dependencies

* pandas
* scikit-learn
* nltk
* streamlit
* joblib

## Notes

* Adjust the dataset path and column names in `train_model.py` if necessary.
* The model's accuracy depends on the quality and size of the training dataset.
* You can improve the model by experimenting with different machine learning algorithms and hyperparameters.

## sample_movie_reviews 
    "An absolute masterpiece! The storyline was captivating and the acting was phenomenal.",
    "I regret watching this movie. It was a complete waste of time.",
    "The film was decent, but nothing extraordinary. Good for a one-time watch.",
    "The plot was predictable and the characters were poorly developed.",
    "An incredible cinematic experience with stunning visuals and powerful performances.",
    "I was bored throughout. The pacing was terribly slow.",
    "A brilliantly directed movie with excellent dialogue and gripping scenes.",
    "Mediocre at best. The jokes fell flat and the script was weak.",
    "I loved every minute of it! A true work of art.",
    "The movie was okay, but the ending was disappointing.",
    "An emotional rollercoaster that had me hooked from start to finish.",
    "Horrible acting, clichéd plot, and terrible dialogue. Just awful.",
    "Visually stunning but lacked substance in the story.",
    "Amazing performances by the entire cast. Highly recommended!",
    "It was just average. Not bad, but not great either."

## Author 
* [Taqi Javed]
