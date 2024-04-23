import random

import numpy as np
from torch.optim import Adam
import torch
from network import FeedForwardNN
from collections import deque
from train_eval import train
from utils import huber_loss


class Agent:

    def __init__(self, obs_dim, action_dim):

        # agent config
        self.obs_dim = obs_dim  # normalized previous days
        self.action_dim = action_dim  # [sit, buy, sell]
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True
        # model config
        self.gamma = 0.95  # affinity for long term reward
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.model = FeedForwardNN(self.obs_dim, self.action_dim)
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def get_action(self, obs, is_eval=False):

        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_dim)

        if self.first_iter:
            self.first_iter = False
            return 1

        action_probs = self.model.predict(obs)
        return np.argmax(action_probs[0])

    def learn(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        x_train, y_train = [], []

        for obs, action, reward, next_obs, done in mini_batch:
            if done:
                target = reward
            else:
                # approximate deep q-learning equation
                next_pred = self.model(next_obs)
                _, predicted = torch.max(next_pred.data, 1)
                target = reward + self.gamma * _

            # estimate q-values based on current state
            q_values = self.model(obs).data
            # update the target for current action based on discounted reward
            q_values[0][action] = target

            x_train.append(obs[0])
            y_train.append(q_values[0].tolist())
        outputs = self.model(torch.FloatTensor(x_train))
        loss = self.loss(outputs, y_train)
        loss.backward()
        self.optimzer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss



