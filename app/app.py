from code.inferencePipeline import inference
import pandas as pd
import config
import tensorflow as tf
import streamlit as st
from typing import Tuple

# code to replace special character from string using regex


def remove_special_characters(text):
    """Remove special characters from a string"""
    import re
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def analyze_sentiment(model: tf.keras.Model, data: pd.DataFrame) -> Tuple[str, float]:
    """
    Analyze sentiment of a review using a trained model.
    Args:
        model (tf.keras.Model): Trained model to use for sentiment analysis.
        data (pd.DataFrame): DataFrame containing the review text to analyze.
    Returns:
        Tuple[str, float]: A tuple containing the sentiment label and the sentiment probability.
        The sentiment label is one of 'Positive', 'Negetive'.
        The sentiment probability is a float between 0 and 1.
    """
    try:
        LABELS = ('Positive', 'Negetive')
        result = inference(model, data)
        label_idx = tf.argmax(result, axis=1)[0]
        label = LABELS[label_idx]
        prob = result[0, label_idx]*100
        return label, prob
    except Exception as e:
        raise e


def run(model):
    st.title("Text Sentiment Analysis")

    # Input text area
    user_input = st.text_area("Enter your review here:")
    data = pd.DataFrame(
        {'reviewText': [remove_special_characters(user_input.lower()),]})

    # Analyze button
    if st.button("Analyze Sentiment"):
        if user_input:
            label, prob = analyze_sentiment(model, data)
            st.success(f" {label} Sentiment: {prob:.2f}%")
        else:
            st.warning("Please enter some text before analyzing.")
