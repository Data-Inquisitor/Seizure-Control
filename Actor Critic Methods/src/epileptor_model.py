import time
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from src.environments import Epileptor
from src.settings import *

import cProfile
import re


class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()

    def _createModel(self):
        model = Sequential()

        model.add(Dense(units=16, activation='relu', input_dim=self.stateCnt))
        model.add(Dense(units=self.actionCnt, activation='linear'))

        opt = RMSprop(lr=LEARNING_RATE)
        model.compile(loss='mse', optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(np.concatenate(s).reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())


class Memory:   # stored as ( s, a, r, s_, a_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity


class Agent(object):

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.epsilon = 0
        self.steps = 0
        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, state, action_df):
        if random.random() < self.epsilon:
            action_num = random.randint(0, self.actionCnt-1)
            return action_df.loc[(action_df['Action'] == action_num), ['Action', 'Frequency', 'Amplitude', 'Cost']]
        else:
            action_num = np.argmax(self.brain.predictOne(state))
            return action_df.loc[(action_df['Action'] == action_num), ['Action', 'Frequency', 'Amplitude', 'Cost']]

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our experience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)
        no_state = np.zeros(self.stateCnt)

        states = np.array([o[0] for o in batch])
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch])

        predictions = self.brain.predict(states)
        future_predictions = self.brain.predict(states_)

        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            states, action, reward, states_, actions_ = o
            unpack_action = action['Action'].values[0]

            t = predictions[i]

            t[unpack_action] = reward + GAMMA * np.amax(future_predictions[i]) - np.amax(predictions[i])

            x[i] = states.reshape(1, 2)

            y[i] = t

        self.brain.train(x, y)


class Environment(object):
    def __init__(self, max_time_steps, time_step, state_estimation, reward_filter_coeff, stim_block_samples,
                 actions_df):
        self.max_time_steps = max_time_steps
        self.stim_block_samples = stim_block_samples
        self.environment = Epileptor(time_step, state_estimation, reward_filter_coeff)
        self.actions_df = actions_df
        self.stim_frequencies = list()
        self.stim_amplitudes = list()
        self.cumulative_rewards = list()

    def run(self, agent, next_action):
        counts = 0
        # Observe initial states
        states, reward = self.environment.get_state_space(next_action['Cost'].values)
        t1 = time.time()
        for step in tqdm(range(int(self.max_time_steps))):

            # Take an action
            action = next_action

            if (np.mod(counts, self.stim_block_samples) == 0) and (counts > 5):
                # If it is time to take a new action...
                # Take a step forward
                self.environment.integrate(action, step)
                # Get new states and the reward for taking action=action
                states_, reward = self.environment.get_state_space(action['Cost'].values,
                                                                   init_state_space=False,
                                                                   init_filter=False)
                # Select a new action
                next_action = agent.act(states_, self.actions_df)

                # Observe new state
                sarsa_packet = (states.reshape(1, 2).flatten(),
                                action,
                                reward,
                                states_.reshape(1, 2).flatten(),
                                next_action)
                agent.observe(sarsa_packet)
                # Learn from past experiences
                agent.replay()
            elif (np.mod(counts, self.stim_block_samples) == 0) and counts < 6:
                # If it is not time to choose a new action and the simulation has begun
                # Take a step forward
                self.environment.integrate(action, step)
                # Observe reward and new states
                states_, reward = self.environment.get_state_space(action['Cost'].values,
                                                                   init_state_space=False,
                                                                   init_filter=True)
            else:
                # If it is not time to choose a new action
                # Take a step forward
                self.environment.integrate(action, step)
                # Observe reward and new states
                states_, reward = self.environment.get_state_space(action['Cost'].values,
                                                                   init_state_space=False,
                                                                   init_filter=False)

            # Update states and add reward to cumulative reward
            states = states_
            self.cumulative_rewards.append(reward)
            self.stim_frequencies.append(action['Frequency'].values[0])
            self.stim_amplitudes.append(action['Amplitude'].values[0])
            counts += 1

        print('Time to complete rounds: {:.3f}'.format(time.time() - t1))


def main():

    therapy_agent = Agent(num_states, num_actions)

    env = Environment(MAX_TIME_STEPS, PERIOD, states, reward_coeff, STIM_BLOCK_SAMPLES,
                      actions_df)

    init_actions = actions_df.loc[np.random.randint(0, num_actions, 1), :]
    env.run(therapy_agent, init_actions)

    time = np.linspace(0, MAX_TIME_STEPS * PERIOD, MAX_TIME_STEPS)
    fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    ax[0].plot(time, env.environment.lfp)
    ax[0].set_ylabel('LFP')
    ax[1].plot(time, env.stim_amplitudes, color='green')
    ax[1].set_ylabel('Stimulation Amplitudes)')
    ax2 = ax[1].twinx()
    ax2.plot(time, env.stim_frequencies, color='red')
    ax2.set_ylabel('Stimulation Frequencies')
    ax[2].plot(time, env.cumulative_rewards, color='black')
    ax[2].set_ylabel('Cumulative Rewards')
    plt.show()


if __name__ in '__main__':
    main()
