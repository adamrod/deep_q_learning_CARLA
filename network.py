from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
import numpy as np


class DQNetwork:
    def __init__(self, action_size, input_shape, logger=None):
        self.action_size = action_size
        self.input_shape = input_shape
        self.logger = logger

        self.learning_rate = 0.001
        self.discount_rate = 0.95

        self.model = self._create_model()
        if logger is not None:
            self.logger.save_network_details(
                {'learning rate: ' : self.learning_rate, 
                'discount_rate: ' : self.discount_rate},
                self.model)
        self.loss_hist = []
        self.acc_hist = []

    def _create_model(self):
        model = Sequential()

        model.add(Conv2D(20, (3, 3), input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(20, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(15))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss="mse", optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model

    def train(self, minibatch):
        inputs = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            inputs.append(state)
            
            target = reward
            if not done:
                target = reward + self.discount_rate * np.amax(self.model.predict(next_state)[0])
            target_all_actions = self.model.predict(state)[0]
            target_all_actions[action] = target

            targets.append(target_all_actions)

        inputs = np.squeeze(np.asarray(inputs), axis=1)
        targets = np.asarray(targets)

        history = self.model.fit(inputs, targets, batch_size=len(minibatch), epochs=1, verbose=0)
        self.loss_hist.append(history.history['loss'][0])
        self.acc_hist.append(history.history['accuracy'][0])

    def predict(self, state):
        return self.model.predict(state)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def plot_model(self):
        import tensorflow as tf
        tf.keras.utils.plot_model(
            self.model,
            to_file='model.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='LR',
            expand_nested=False,
            dpi=300
        )
