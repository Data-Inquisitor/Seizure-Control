import time
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

from src.environments import Epileptor
from src.settings import *
from src.SumTree import SumTree


class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.is_weight = 1

        self.model = self._createModel()
        self.model_ = self._createModel()

    # TODO: Implement Importance sampling loss function
    def _importance_sampling_loss(self):

        return None

    def _createModel(self):

        model = Sequential()

        for layer in range(NUM_HIDDEN_LAYERS):
            model.add(Dense(units=NUM_UNITS_PER_LAYER, activation='relu', input_dim=self.stateCnt))
        model.add(Dense(units=self.actionCnt, activation='linear'))

        opt = Adam(lr=LEARNING_RATE)
        model.compile(loss='mean_squared_error', optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=32, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(np.concatenate(s).reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())


class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        # Proportional prioritization
        # epsilon is designed to make sure no transition has zero priority
        # alpha controls the difference between high and low error. If alpha=0 then all experiences are equal.
        return (error + PER_EPSILON) ** PER_ALPHA

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


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
            return action_df.loc[(action_df['Action'] == action_num), :]
        else:
            action_num = np.argmax(self.brain.predictOne(state))
            return action_df.loc[(action_df['Action'] == action_num), :]

    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self.get_targets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def get_targets(self, batch):
        no_state = np.zeros(self.stateCnt)
        batch_len = len(batch)
        states = np.array([o[1][0] for o in batch])
        states_ = np.array([(no_state if o[1][3] is None else o[1][3]) for o in batch])

        p = self.brain.predict(states)

        # Learn two Q-functions
        p_ = self.brain.predict(states_, target=False)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batch_len, self.stateCnt))
        y = np.zeros((batch_len, self.actionCnt))
        errors = np.zeros(len(batch))

        for i in range(len(batch)):
            o = batch[i][1]
            states_, action, reward, next_states_ = o
            unpack_action = action['Action'].values[0]

            t = p[i]
            current_qval = t[unpack_action]

            # Double Q-learning update
            t[unpack_action] = reward + GAMMA * pTarget_[i][np.argmax(p_[i])]  # double DQN

            # Double Q-learning error
            errors[i] = abs(t[unpack_action] - current_qval)

            x[i] = states_.reshape(1, self.stateCnt)
            y[i] = t

        return x, y, errors

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self.get_targets(batch)

        # update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)


class Environment(object):
    def __init__(self, max_time_steps, time_step, state_estimation, stim_block_samples, action_selection_df):
        self.max_time_steps = max_time_steps
        self.stim_block_samples = stim_block_samples
        self.environment = Epileptor(time_step, state_estimation)
        self.action_selection_df = action_selection_df
        self.stim_frequencies = list()
        self.stim_amplitudes = list()
        self.cumulative_rewards = list()

    def run(self, agent, next_action):
        # Observe initial states
        states_, reward = self.environment.get_state_space(next_action['Cost'].values)
        t1 = time.time()
        for step in tqdm(range(int(self.max_time_steps))):

            # Take an action
            action = next_action

            if (np.mod(step, self.stim_block_samples) == 0) and (step > 5):
                # If it is time to take a new action...
                # Take a step forward
                self.environment.integrate(action, step)
                # Get new states and the reward for taking action=action
                next_states, reward = self.environment.get_state_space(action['Cost'].values,
                                                                       init_state_space=False,
                                                                       init_filter=False)
                # Select a new action
                next_action = agent.act(next_states, self.action_selection_df)

                # Observe new state
                q_packet = (states_.reshape(1, agent.stateCnt).flatten(),
                            action,
                            reward,
                            next_states.reshape(1, agent.stateCnt).flatten())

                agent.observe(q_packet)
                # Learn from past experiences
                agent.replay()
            elif (np.mod(step, self.stim_block_samples) == 0) and step < 6:
                # If it is not time to choose a new action and the simulation has begun
                # Take a step forward
                self.environment.integrate(action, step)
                # Observe reward and new states
                next_states, reward = self.environment.get_state_space(action['Cost'].values,
                                                                       init_state_space=False,
                                                                       init_filter=True)
            else:
                # If it is not time to choose a new action
                # Take a step forward
                self.environment.integrate(action, step)
                # Observe reward and new states
                next_states, reward = self.environment.get_state_space(action['Cost'].values,
                                                                       init_state_space=False,
                                                                       init_filter=False)

            # Update states and add reward to cumulative reward
            states_ = next_states
            self.cumulative_rewards.append(reward)
            self.stim_frequencies.append(action['Frequency'].values[0])
            self.stim_amplitudes.append(action['Amplitude'].values[0])

        print('Time to complete rounds: {:.3f}'.format(time.time() - t1))


def main():

    therapy_agent = Agent(num_states, num_actions)

    env = Environment(MAX_TIME_STEPS, PERIOD, states, STIM_BLOCK_SAMPLES,
                      actions_df)

    init_actions = actions_df.loc[np.random.randint(0, num_actions, 1), :]
    env.run(therapy_agent, init_actions)

    t = np.linspace(0, MAX_TIME_STEPS * PERIOD, MAX_TIME_STEPS)
    fig, ax = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    ax[0].plot(t, env.environment.lfp, label='LFP')
    ax[0].plot(t, env.environment.filter_state_1[:-1], label='filter_state1')
    ax[0].plot(t, env.environment.filter_state_2[:-1], label='filter_state2')
    ax[0].plot(t, env.environment.filter_state_3[:-2], label='filter_state3')
    ax[0].set_ylabel('A.U.')
    ax[0].legend()
    ax[2].plot(t, env.stim_amplitudes, color='green', label='Amplitude')
    ax[2].set_ylabel('Stimulation Amplitudes')
    ax[2].legend()
    ax3 = ax[2].twinx()
    ax3.plot(t, env.stim_frequencies, color='red', label='Frequency')
    ax3.set_ylabel('Stimulation Frequencies')
    ax3.legend()
    ax[3].plot(t, env.cumulative_rewards, color='black')
    ax[3].set_ylabel('Cumulative Rewards')
    plt.show()


if __name__ in '__main__':
    main()
