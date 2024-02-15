from app import app
import tensorflow as tf
import config
import sys, os
current = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append(os.path.join(parent, 'code'))
sys.path.append(os.path.join(parent, 'app'))
sys.path.append(os.path.join(parent, 'test'))

if __name__ == "__main__":
    
    # Load the saved model
    model = tf.keras.models.load_model(config.MODEL_PATH)

    app.run(model=model)