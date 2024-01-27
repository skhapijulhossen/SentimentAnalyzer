import sys
import os

current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import tensorflow as tf
import pandas as pd
import logging
import config
import joblib


def inference(model: tf.keras.Model, data: pd.DataFrame):
    """
    :param model:
    :param data:
    :return:
    """
    try:
        logging.info("Start inference")
        # Load tokenizer using joblib
        tokenizer = joblib.load(config.TOKENIZER_PATH)
        # Preprocess the data
        data = tokenizer.texts_to_sequences(data.reviewText)
        data = tf.keras.preprocessing.sequence.pad_sequences(
            data, maxlen=100, padding='post', truncating='post')

        # inference using model
        prob = model.predict(data)
        logging.info(f"Inference completed: {prob[0]}")

        return prob
    except Exception as e:
        logging.error(e)
        raise e
