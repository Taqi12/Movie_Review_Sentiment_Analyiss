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
