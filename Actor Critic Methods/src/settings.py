import math
from scipy.signal import butter
import numpy as np
import pandas as pd

"""################### Epileptor Settings ###################"""
# Initialize parameters
x0 = 2.0      # If this value is above 2.92 seizures will not emerge
y0 = 1
taux = 0.5
tau0 = 50    # This time constant controls how often seizures occur
tau1 = 0.5
tau2 = 2
Irest1 = 3.1
Irest2 = 0.45
Ep_gamma = 0.01

"""################### Simulation Settings ####################"""
TOTAL_TIME = 1000  # How long we want to run the simulation (in seconds)
PERIOD = 0.002     # Sampling period in seconds
Fs = 1 / PERIOD    # Sampling rate
MAX_TIME_STEPS = TOTAL_TIME / PERIOD  # Maximum number of iterations to loop through
STIM_BLOCK_TIME = 15  # How often to choose a different action (in seconds)
STIM_BLOCK_SAMPLES = STIM_BLOCK_TIME * Fs  # How often to choose a different action (in samples)

"""################### DNN Settings ####################"""
NUM_HIDDEN_LAYERS = 3
NUM_UNITS_PER_LAYER = 64
MEMORY_CAPACITY = 200000  # How many samples of previous states to hold in memory buffer
BATCH_SIZE = 32  # How many samples to train on
LEARNING_RATE = 0.1  # Learning rate of backpropagation algorithm
UPDATE_TARGET_FREQUENCY = 10000  # How often to update weights

"""################### RL Settings ####################"""
GAMMA = 0.005    # Discounting factor, how much do we want to trust in the future
MAX_EPSILON = 0.75  # Maximum probability of choosing a random action
MIN_EPSILON = 0.01  # Minimum probability of choosing a random action
LAMBDA = 0.01  # speed of decay of probability of choosing a random action
COST_WEIGHT = 1  # How much we want to weight the cost of stimulation therapy

# Exponential filter for the reward signal
TAU_FILT = math.exp(-1./4000)
TAU_NORM = 1/(1-TAU_FILT)

"""################### PER Settings ####################"""
PER_EPSILON = 0.01
PER_ALPHA = 0.5

"""################### Filter Settings ####################"""
# Filter coefficients to get states from LFP
b_coeff_h, a_coeff_h = butter(N=1, Wn=10 / (Fs / 2), btype='hp', fs=Fs)
b_coeff_l, a_coeff_l = butter(N=1, Wn=1 / (Fs / 2), btype='low', fs=Fs)
b_coeff_m1, a_coeff_m1 = butter(N=1, Wn=[1 / (Fs / 2), 4 / (Fs / 2)], btype='bp', fs=Fs)
b_coeff_m2, a_coeff_m2 = butter(N=1, Wn=[4 / (Fs / 2), 10 / (Fs / 2)], btype='bp', fs=Fs)

states = {'State1': {'Num': b_coeff_h, 'Den': a_coeff_h},
          'State2': {'Num': b_coeff_l, 'Den': a_coeff_l},
          'State3': {'Num': b_coeff_m1, 'Den': a_coeff_m1},
          'State4': {'Num': b_coeff_m2, 'Den': a_coeff_m2}}

"""################### Action Settings ####################"""
actions_df = pd.DataFrame(columns=['Action', 'Frequency', 'Amplitude'])
actions_df['Action'] = np.arange(0, 5)
actions_df['Frequency'] = [30, 35, 40, 45, 50]
NO_STIM = True
if NO_STIM is True:
    actions_df['Amplitude'] = [0.0]*5
else:
    actions_df['Amplitude'] = [-1] * 5
actions_df['Cost'] = actions_df['Frequency'] * (actions_df['Amplitude'] ** 2)
num_states = len(states.keys())
num_actions = actions_df.shape[0]