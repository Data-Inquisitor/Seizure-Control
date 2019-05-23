"""
This tutorial shows how a neural network (or Deep Neural Network) can be used to approximate the expected value of
different state-action pairs in a computational model of seizures. This script uses an Off-Policy Q-Learning paradigm to
determine optimal stimulation parameters to suppress seizures in the Epileptor computational model.

author: Vivek Nagaraj, 2019

Special thanks to Jaromir Janish for the Deep-Q-Network tutorial he put together. His code provided a fantastic
starting place for designing my reinforcment learning experiments.
Link: https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/

Epileptor Model
1) Jirsa VK, Stacey WC, Quilichini PP, Ivanov AI, Bernard C. On the nature of seizure dynamics.
Brain. 2014 Jun 10;137(8):2210-30.
2) Proix T, Bartolomei F, Chauvel P, Bernard C, Jirsa VK. Permittivity coupling across brain regions determines seizure
recruitment in partial epilepsy. Journal of Neuroscience. 2014 Nov 5;34(45):15009-21.
"""


import os
import random
from time import localtime, strftime

from src.epileptor_model import *

WORK_DIR = r'C:\Users\vnaga\Google Drive\Seizure-Control\Actor Critic Methods'
FIGURE_DIR = os.path.join(WORK_DIR, 'figures')

random.seed(1)


def visualizations(time_s, lfp, s1, s2, s3, s4, amp, freq, rew, output_dir):
    fig, ax = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    ax[0].plot(time_s, lfp, color='black')
    ax[0].set_ylabel('LFP (A.U.)')
    ax[0].set_title('Local Field Potential')
    ax[1].plot(time_s, s1[:-1], label='State1')
    ax[1].plot(time_s, s2[:-1], label='State2')
    ax[1].plot(time_s, s3[:-2], label='State3')
    ax[1].plot(time_s, s4[:-2], label='State4')
    ax[1].set_ylabel('State Variables (A.U.)')
    ax[1].legend()
    ax[2].plot(time_s, amp, color='green', label='Amplitude')
    ax[2].set_ylabel('Stimulation Amplitudes')
    ax[2].legend(loc='upper left')
    ax2 = ax[2].twinx()
    ax2.plot(time_s, freq, color='red', label='Frequency')
    ax2.set_ylabel('Stimulation Frequencies')
    ax2.legend(loc='upper left')
    ax[3].plot(time_s, np.cumsum(rew), color='black')
    ax[3].set_ylabel('Cumulative Rewards (A.U.)')
    ax[3].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    str_time = strftime("%Y-%m-%d_%H%M%S", localtime())
    fig.savefig(os.path.abspath(os.path.join(output_dir, str_time + '_DNNRL_results.png')))


def run():
    therapy_agent = Agent(num_states, num_actions)

    env = Environment(MAX_TIME_STEPS, PERIOD, states, STIM_BLOCK_SAMPLES,
                      actions_df)

    init_actions = actions_df.loc[np.random.randint(0, num_actions, 1), :]
    env.run(therapy_agent, init_actions)

    time_vec = np.linspace(0, MAX_TIME_STEPS * PERIOD, MAX_TIME_STEPS)

    visualizations(time_vec, env.environment.lfp, env.environment.filter_state_1,
        env.environment.filter_state_2, env.environment.filter_state_3, env.environment.filter_state_4,
        env.stim_amplitudes, env.stim_frequencies, env.cumulative_rewards, FIGURE_DIR)


if __name__ in '__main__':
    run()