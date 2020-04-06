from network import DQNetwork
import numpy as np
from env import CarlaEnv
from collections import deque
from utils import prepare_environment, read_pickle, get_mean_max_q_values
from logger import Logger
import random
import time


class DQNAgent:
    def __init__(self, state_size, action_size, logger=None, epsilon=1.0,
                 epsilon_min=0.1, epsilon_decay=0.997, q_maxlen=5000):
        self.state_size = state_size
        self.action_size = action_size
        self.logger = logger
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=q_maxlen)
        self.dqnetwork = DQNetwork(action_size, state_size, logger)
        
    def take_action(self, state):
        if np.random.rand() <= self.epsilon:
             return np.random.randint(self.action_size)
        actions_q_values = self.dqnetwork.predict(state)
        return np.argmax(actions_q_values[0])
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        if len(self.memory) >= batch_size:
            minibatch = random.sample(self.memory, batch_size)
            self.dqnetwork.train(minibatch)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def exit(self):
        self.logger.save_fit_history(self.dqnetwork.loss_hist, self.dqnetwork.acc_hist)
        self.logger.save_model_weights(self.dqnetwork.model, "_exit")


if __name__ == '__main__':
    prepare_environment()

    logger = Logger()
    env = CarlaEnv(spawn_index=0, action_size=5)
    state = env.reset()
    agent = DQNAgent(state.shape, 5, logger)

    try:
        max_episodes = 10000
        max_frames = 5000
        total_rewards = []
        epsilons = []
        mean_max_q_values = []

        for episode in range(max_episodes):
            spawn_index = random.randint(0, len(env.spawn_points) - 1)
            state = env.reset(spawn_index)
            state = np.expand_dims(state, axis=0)
            total_reward = 0
            epsilons.append(agent.epsilon)
            for frame in range(max_frames):
                action = agent.take_action(state)
                next_state, reward, done = env.step(action)
                total_reward += reward
                next_state = np.expand_dims(next_state, axis=0)
                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if done or frame == max_frames - 1:
                    total_rewards.append(total_reward)
                    logger.episode_summary(episode, frame + 1, total_reward, agent.epsilon)
                    break

            agent.train(50)
            agent.update_epsilon()
    finally:
        env.close()
        agent.exit()
        logger.save_total_rewards(total_rewards)
        logger.save_epsilons(epsilons)
