import math
from scipy.signal import butter
import numpy as np
import pandas as pd

"""################### Epileptor Settings ###################"""
# Initialize parameters
x0 = 2.0
y0 = 1
taux = 0.5
tau0 = 100
tau1 = 0.5
tau2 = 2
Irest1 = 3.1
Irest2 = 0.45
Ep_gamma = 0.01

"""################### Simulation Settings ####################"""
TOTAL_TIME = 1000  # In seconds
PERIOD = 0.002
Fs = 1 / PERIOD
MAX_TIME_STEPS = TOTAL_TIME / PERIOD
DO_NOT_STIM = True
STIM_BLOCK_TIME = 20  # In seconds
STIM_BLOCK_SAMPLES = STIM_BLOCK_TIME * Fs

"""################### DNN Settings ####################"""
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64
LEARNING_RATE = 0.01

"""################### RL Settings ####################"""
GAMMA = 0.01
MAX_EPSILON = 0.4
MIN_EPSILON = 0.01
LAMBDA = 0.001  # speed of decay
COST_WEIGHT = 1
UPDATE_TARGET_FREQUENCY = 100

# Exponential filter for the reward signal
TAU_FILT = math.exp(-1./1000)
TAU_NORM = 1/(1-TAU_FILT)

"""################### Filter Settings ####################"""
b_coeff_h, a_coeff_h = butter(N=1, Wn=0.1 / (Fs / 2), btype='hp', fs=Fs)
b_coeff_l, a_coeff_l = butter(N=1, Wn=0.1 / (Fs / 2), btype='low', fs=Fs)
b_coeff_rew, a_coeff_rew = butter(N=1, Wn=[0.1 / (Fs / 2), 4 / (Fs / 2)], btype='bp', fs=Fs)
reward_coeff = {'Num': b_coeff_rew, 'Den': a_coeff_rew}

states = {'State1': {'Num': b_coeff_h, 'Den': a_coeff_h},
          'State2': {'Num': b_coeff_l, 'Den': a_coeff_l}}

"""################### Action Settings ####################"""
actions_df = pd.DataFrame(columns=['Action', 'Frequency', 'Amplitude'])
actions_df['Action'] = np.arange(0, 5)
actions_df['Frequency'] = [10, 15, 20, 25, 30]
#actions_df['Amplitude'] = [0.0] + [-0.1, -0.2, -0.4] * 3
actions_df['Amplitude'] = [-0.094]*5
#actions_df['Amplitude'] = [0.0] * 10
actions_df['Cost'] = actions_df['Frequency'] * (actions_df['Amplitude'] ** 2)
num_states = len(states.keys())
num_actions = actions_df.shape[0]