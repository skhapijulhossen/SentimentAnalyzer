import sys
import os

current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)


from featurePipeline import extract
import logging
import pandas as pd
import config
from sklearn.model_selection import train_test_split
import tensorflow as tf
from joblib import dump, load
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras import Sequential, callbacks



def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by extracting features and dropping unnecessary columns.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    logging.info("Preprocessing data...")
    try:
        data = extract(config.TRAIN_DATA_SOURCE)
        data = data[['overall', 'reviewText']]

        # Label Sentiment
        data.loc[:, 'sentiment'] = 0
        data.loc[data['overall'] < 3, 'sentiment'] = 1

        # Spliting
        data = pd.concat([data[data.sentiment == 0].head(
            config.LENGTH), data[data.sentiment == 1].head(config.LENGTH)], ignore_index=True)

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            data['reviewText'], data['sentiment'], test_size=0.25, random_state=42)

        # Encode Labels
        y_train = tf.one_hot(y_train, depth=2)
        y_test = tf.one_hot(y_test, depth=2)
        
        # Get Numerical Representation
        X_train = X_train.to_list()

        # init Tokenizer
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=config.VOCAB_SIZE, oov_token='')

        # fit on train
        tokenizer.fit_on_texts(X_train)

        # Generate sequence
        train_tokens = tokenizer.texts_to_sequences(X_train)
        train_padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(
            train_tokens, maxlen=100, padding='post', truncating='post')

        # Process Test Data
        test_tokens = tokenizer.texts_to_sequences(X_test)
        test_padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(
            test_tokens, maxlen=100, padding='post', truncating='post')

        # Save tokenizer
        dump(tokenizer, config.TOKENIZER_PATH)

        return train_padded_tokens, test_padded_tokens, y_train, y_test

    except Exception as e:
        logging.error(e)
        raise e


def trainModel(data: pd.DataFrame) -> None:
    """
    Train a machine learning model on the given data.

    Args:
        data (pd.DataFrame): The input data.

    Returns:
        None
    """
    logging.info("Training model...")
    device = 'cuda' if tf.test.is_gpu_available() else 'cpu'
    try:
        # Preprocess data
        train_padded_tokens, test_padded_tokens, y_train, y_test = preprocess(
            data)

        # Create model
        lstm_model = Sequential(
            [
                Embedding(input_dim=config.VOCAB_SIZE, output_dim=128),
                Bidirectional(
                    LSTM(64, return_sequences=True)
                ),
                Bidirectional(
                    LSTM(128)
                ),
                Dense(128, activation='relu'),
                Dropout(rate=0.5),
                Dense(128, activation='relu'),
                Dropout(rate=0.25),
                Dense(64, activation='relu'),
                Dense(2, activation='softmax')
            ]
        )

        # EarlyStopping
        es = callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max', verbose=1, patience=3)

        # Train
        with tf.device(device):
            # Compile Model
            lstm_model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy',]
            )

            # Train and Validate
            lstm_model.fit(
                train_padded_tokens, y_train,
                epochs=config.EPOCHS, steps_per_epoch=config.STEPS_PER_EPOCH,
                validation_data=(test_padded_tokens, y_test),
                validation_steps=50, callbacks=[es]
            )

        # Save Model
        lstm_model.save(config.MODEL_PATH)

    except Exception as e:
        logging.error(e)
        raise e




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting Train pipeline")
    trainModel(pd.DataFrame())
    logging.info("Train pipeline completed successfully")