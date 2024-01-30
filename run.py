from app import app
import tensorflow as tf
import config

if __name__ == "__main__":
    
    # Load the saved model
    model = tf.keras.models.load_model(config.MODEL_PATH)

    app.run(model=model)