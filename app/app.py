from code.inferencePipeline import inference
import pandas as pd
import config
import tensorflow as tf
import streamlit as st
import sys
import os

current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append(os.path.join(parent, 'code'))


# code to replace special character from string using regex


def remove_special_characters(text):
    """Remove special characters from a string"""
    import re
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def run(model):
    st.title("Text Sentiment Analysis")

    # Input text area
    user_input = st.text_area("Enter your review here:")
    data = pd.DataFrame(
        {'reviewText': [remove_special_characters(user_input.lower()),]})

    # Analyze button
    if st.button("Analyze Sentiment"):
        LABELS = ('Positive', 'Negetive')
        if user_input:
            result = inference(model, data)
            label_idx = tf.argmax(result, axis=1)[0]
            label = LABELS[label_idx]
            prob = result[0, label_idx]*100
            st.success(f" {label} Sentiment: {prob:.2f}%")
        else:
            st.warning("Please enter some text before analyzing.")
