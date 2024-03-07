# test_sentiment_analyzer.py

from code.inferencePipeline import inference
from app.app import remove_special_characters, analyze_sentiment
import pandas as pd


def test_analyze_sentiment_positive():
    text = "I love this product!"
    data = pd.DataFrame(
        {'reviewText': [remove_special_characters(text.lower()),]})

    assert analyze_sentiment(text)[0] == "Positive"


def test_analyze_sentiment_negative():
    text = "This product is terrible."
    data = pd.DataFrame(
        {'reviewText': [remove_special_characters(text.lower()),]})
    assert analyze_sentiment(text)[0] == "Negative"
