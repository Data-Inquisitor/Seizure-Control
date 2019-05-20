import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.settings import *


class Epileptor(object):
    """

    """
    def __init__(self, time_step, state_estimators):
        """
        Initialize parameteres and state vectors
        """

        self.state_x1 = 0.022
        self.state_y1 = 0.91
        self.state_z = 3.84
        self.state_x2 = -1.11
        self.state_y2 = 0.73
        self.state_u = 0

        self.time_step = time_step
        self.past_step = 0
        self.scale_factor = 2*np.sqrt(self.time_step)

        self.lfp = list()
        self.z_states = list()
        self.filter_state_1 = list()
        self.filter_state_2 = list()
        self.filter_state_3 = [0]
        self.smoothLFPPower = list()

        self.state_estimators = state_estimators

    def integrate(self, action, step):
        def _output_1(x1, x2, z):
            if x1 >= 0:
                return (x2 - 0.6 * (z - 4)**2) * x1
            else:
                return x1**3 - 3 * x1**2

        def _output_2(x2):
            if x2 >= -0.25:
                return 6 * (x2 + 0.25)
            else:
                return 0

        self.state_x1 = self.state_x1 + self.time_step * \
                                            (self.state_y1 - _output_1(self.state_x1, self.state_x2, self.state_z) -
                                             self.state_z + Irest1) / taux

        self.state_x2 = self.state_x2 + self.time_step * (-self.state_y2 + self.state_x2 - self.state_x2**3 +
                                                              Irest2 + 3 * self.state_u - 0.3 *
                                                              (self.state_z - 3.5)) / taux + \
                            0.025 * np.random.randn() * self.scale_factor

        h = x0 + (10 / (1 + np.exp((-self.state_x1 - 0.5) / 0.1)))
        self.state_z = self.state_z + (self.time_step / tau0) * (h - self.state_z)
        self.state_u = self.state_u - self.time_step * Ep_gamma * (self.state_u - 0.1 * self.state_x1) /\
                                          taux

        self.state_y1 = self.state_y1 + (self.time_step / tau1) * \
                                            (y0 - 5 * (self.state_x1**2) - self.state_y1)
        self.state_y2 = self.state_y2 + (self.time_step / tau2) * \
                                            (-self.state_y2 + _output_2(self.state_x2)) + \
                            0.025 * np.random.randn() * self.scale_factor

        # Add stimulation to x1 and x2 state variables when the steps has exceeded the number of samples between pulses
        if (step - self.past_step) > (1/action['Frequency'].values[0] * (1/self.time_step)):
            self.state_x1 += action['Amplitude'].values
            self.state_x2 += action['Amplitude'].values
            self.past_step = step

        self.lfp.append(-self.state_x1 + self.state_x2)
        self.z_states.append(self.state_z)

    def get_state_space(self, cost, init_state_space=True, init_filter=False):
        if init_state_space is True:
            new_state1 = 0
            new_state2 = 0
            new_state3 = 0
            reward = 0
        elif init_filter is True:
            new_state1 = self.lfp[-1]
            new_state2 = self.lfp[-1]
            new_state3 = self.lfp[-1]
            self.smoothLFPPower.append(self.lfp[-1]**2)
            reward = 0
        else:
            # State space filters
            new_state1 = [self.state_estimators['State1']['Num'][0]*self.lfp[-1] +
                         self.state_estimators['State1']['Num'][1]*self.lfp[-2] -
                         self.state_estimators['State1']['Den'][1]*self.filter_state_1[-1]][0]

            new_state2 = [self.state_estimators['State2']['Num'][0]*self.lfp[-1] +
                         self.state_estimators['State2']['Num'][1]*self.lfp[-2] -
                         self.state_estimators['State2']['Den'][1]*self.filter_state_2[-1]][0]
            if len(self.lfp) == 2:
                new_state3 = self.lfp[-2]
            else:
                new_state3 = [self.state_estimators['State3']['Num'][0]*self.lfp[-1] +
                             self.state_estimators['State3']['Num'][1]*self.lfp[-2] +
                              self.state_estimators['State3']['Num'][2] * self.lfp[-3] -
                             self.state_estimators['State3']['Den'][1]*self.filter_state_3[-1] -
                              self.state_estimators['State3']['Den'][2] * self.filter_state_3[-2]][0]

            # Reward filter
            filtLFP = np.ma.average([i ** 2 for i in self.lfp[-5:]])

            self.smoothLFPPower.append(filtLFP**2 / TAU_NORM + TAU_FILT * self.smoothLFPPower[-1])

            reward = -np.log(self.smoothLFPPower[-1]) - cost*COST_WEIGHT

        self.filter_state_1.append(new_state1)
        self.filter_state_2.append(new_state2)
        self.filter_state_3.append(new_state3)

        return np.asarray([self.filter_state_1[-1], self.filter_state_2[-1], self.filter_state_2[-1]]), reward


if __name__ in '__main__':
    # Open loop stimulation
    environment = Epileptor(PERIOD, states)
    for step in tqdm(range(int(MAX_TIME_STEPS))):
        if step > 5:
            environment.integrate(actions_df.loc[actions_df.Action == 3, :], step)
            environment.get_state_space(0, init_state_space=False, init_filter=False)
        if step < 6:
            environment.integrate(actions_df.loc[actions_df.Action == 3, :], step)
            environment.get_state_space(0, init_state_space=False, init_filter=True)
        else:
            environment.integrate(actions_df.loc[actions_df.Action == 3, :], step)
            environment.get_state_space(0, init_state_space=False, init_filter=False)

    time = np.linspace(0, MAX_TIME_STEPS * PERIOD, MAX_TIME_STEPS)
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    ax[0].plot(time, environment.lfp)
    ax[0].set_ylabel('LFP')
    ax[1].plot(time, environment.filter_state_1, color='blue', label='State1')
    ax[1].plot(time, environment.filter_state_2, color='green', label='State2')
    ax[1].plot(time, environment.filter_state_3, color='red', label='State3')
    ax[1].legend()
    plt.show()

