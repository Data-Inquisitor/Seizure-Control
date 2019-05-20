from src.epileptor_model import *


def visualizations(time_s, lfp, s1, s2, s3, amp, freq, rew):
    fig, ax = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    ax[0].plot(time_s, lfp, color='black')
    ax[0].set_ylabel('LFP')
    ax[1].plot(time_s, s1[:-1], label='State1')
    ax[1].plot(time_s, s2[:-1], label='State2')
    ax[1].plot(time_s, s3[:-2], label='State3')
    ax[1].set_ylabel('A.U.')
    ax[1].legend()
    ax[2].plot(time_s, amp, color='green', label='Amplitude')
    ax[2].set_ylabel('Stimulation Amplitudes')
    ax[2].legend()
    ax2 = ax[2].twinx()
    ax2.plot(time_s, freq, color='red', label='Frequency')
    ax2.set_ylabel('Stimulation Frequencies')
    ax2.legend()
    ax[3].plot(time_s, rew, color='black')
    ax[3].set_ylabel('Cumulative Rewards')
    plt.show()


def run():
    therapy_agent = Agent(num_states, num_actions)

    env = Environment(MAX_TIME_STEPS, PERIOD, states, STIM_BLOCK_SAMPLES,
                      actions_df)

    init_actions = actions_df.loc[np.random.randint(0, num_actions, 1), :]
    env.run(therapy_agent, init_actions)

    time = np.linspace(0, MAX_TIME_STEPS * PERIOD, MAX_TIME_STEPS)

    visualizations(time, env.environment.lfp, env.environment.filter_state_1,
        env.environment.filter_state_2, env.environment.filter_state_3,
        env.stim_amplitudes, env.stim_frequencies, env.cumulative_rewards)


if __name__ in '__main__':
    run()