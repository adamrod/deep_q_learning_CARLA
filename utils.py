from keras.models import model_from_json
import numpy as np
import random
import pickle
import cv2

def prepare_environment():
    np.random.seed(1)
    random.seed(1)

    from tensorflow import ConfigProto
    from tensorflow import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

def read_pickle(filepath):
    with open(str(filepath), 'rb') as f:
        return pickle.load(f)

def get_mean_max_q_values(model, test_states):
    return np.mean([np.max(model.predict(state)[0]) for state in test_states])
