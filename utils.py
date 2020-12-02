import matplotlib.pyplot as plt
import numpy as np
import virl


def execute_policy(policy, env):
    """Execute a policy

    :param policy a policy to run. A policy is a callable that takes a state as input and returns an array of probabilities of actions. If the policy returns a tuple the first element is taken.
    :param env the environment to run (a correctly instantiated problem).

    :returns a tuple (states, rewards, action_taken)
    """
    s = env.reset()

    states = [s]
    rewards = []
    action_taken = []
    done = False

    while not done:
        s = states[-1]
        action_probs = policy(s)
        if type(action_probs) == tuple:
            action_probs = action_probs[0]
        action_id = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        s, r, done, i = env.step(action=action_id)

        states.append(s)
        rewards.append(r)
        action_taken.append(action_id)

    return states, rewards, action_taken

def plot_avg_reward(avg_episode_rewards, smoothing_window=5):
    avg_episode_rewards = np.array(avg_episode_rewards)
    smoothed = np.convolve(avg_episode_rewards, np.ones(smoothing_window)/smoothing_window, mode='valid')

    fig, ax = plt.subplots()
    ax.plot(smoothed)
    ax.set_title('Reward per episode')
    ax.set_xlabel('episode')
    ax.set_ylabel('mean reward r(t)')

def plot(states, rewards, action_taken=None, axes=None, problem_id=0):
    """Plot the state/reward diagram after a policy is executed

    :param states the states to use
    :param rewards the rewards to use
    :param action_taken if provided, show the actions performed in the reward plot graph
    :param axes if provided, use the given axes rather than using custom defined axes.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    total_reward = np.sum(rewards)

    labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
    states = np.array(states)
    for i in range(4):
        axes[0].plot(states[:,i], label=labels[i]);

    axes[0].set_title(f"Statistics of the epidemic for problem {problem_id}")
    axes[0].set_xlabel('weeks since start of epidemic')
    axes[0].set_ylabel('State s(t)')
    axes[0].legend()
    axes[1].plot(rewards);
    axes[1].set_title(f'Reward, total reward = {total_reward:.3f}')
    axes[1].set_xlabel('weeks since start of epidemic')
    axes[1].set_ylabel('reward r(t)')

    if action_taken:
        colors = ['r', 'g', 'b', 'k']
        for i in range(4):
            axes[1].vlines(np.where(np.array(action_taken) == i), ymin = np.max(np.min(rewards) -0.050), ymax=0.00, colors=colors[i], linestyle='dashed')

    # print('total reward', total_reward)


def evaluate(policy, problem_id=0, full_eval=False, verbose=True, noisy=False):
    """
    Evaluate a policy

    :param policy a callable that, given in input a state, returns an action
    :param full_eval whether to fully evaluate the policy on all the problems or the first problem only
    :param verbose whether to get verbose output
    :param noisy whether to simulate a noisy environment
    """
    #trained_policy = create_policy(approximator_dl, 0, 4)
    
    if not full_eval:
        limit = 1
        envs = [virl.Epidemic(problem_id=problem_id, noisy=noisy)]
    else:
        limit = 10
        envs = [virl.Epidemic(problem_id=i, noisy=noisy) for i in range(limit)]

    fig, axes = plt.subplots(limit, 2, figsize=(20, 8*limit))

    total_rewards = []

    for i, env in enumerate(envs, start=problem_id):
        states, rewards, action_taken = execute_policy(policy, env)
        if verbose:
            print(i, action_taken)
        # small hack to change the first key from i to 0
        if limit == 1:
            axes_wrapper = [axes[0], axes[1]]
        else:
            axes_wrapper = axes[i]
        plot(states, rewards, action_taken, axes=axes_wrapper, problem_id=i)
        total_rewards.append(sum(rewards))

    if limit > 1:
        _, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.bar(np.arange(limit), total_rewards)
        ax.set_xticks(np.arange(limit))
        return total_rewards

def latex_table(array, row_labels, include_means=False):
    def to_str(f):
        return str(round(float(f), 2))
    table_str = ""
    best_cols = np.argmin(np.array(array), axis=0)
    means = np.mean(np.array(array), axis=1)
    max_mean_index = np.argmin(means)
    for i, row in enumerate(array):
        row_str = f"{row_labels[i]}"
        for j, el in enumerate(row):
            if i == best_cols[j]:
                row_str += " & \\textbf{" + to_str(el) + "}"
                continue
            row_str += " & " + to_str(el)
        if include_means:
            table_str += row_str + " & " + (to_str(means[i]) if i != max_mean_index else "\\textbf{\\underline{" + to_str(means[i]) + "}}") + "\\\ \n"
            continue
        table_str += row_str + "\\\ \n"
    return table_str

def evaluate_stochastic(policy, num_tries=10, noisy=True):
    """
    Evaluate a policy in a stochastic environment.

    horribly copied from generate_readme_plots.ipynb

    :param policy a callable that returns a probability distribution of probabilities
    :param num_tries the number of tries to perform

    """

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(num_tries):
        env = virl.Epidemic(stochastic=True, noisy=noisy)
        states, rewards, actions_taken = execute_policy(policy, env)
        ax.plot(np.array(states)[:,1], label=f'draw {i}')
    ax.set_xlabel('weeks since start of epidemic')
    ax.set_ylabel('Number of Infectious persons')
    ax.set_title(f'Simulation of {num_tries} stochastic episodes without intervention')
    ax.legend()


def policy_greedy(state):
    def eval_reward(state, action):
        policy_severity_factor = 1e11
        a = state[1] + state[2]
        b = (1 - action)

        expected_a = a*(1 + action - 0.1)
        val = (-expected_a - expected_a ** 2 - policy_severity_factor*b - policy_severity_factor*b**2) / policy_severity_factor

        return val

    env = virl.Epidemic()
    
    greedy_rewards = np.array([eval_reward(state, a) for a in env.actions])
    action_id = np.argmax(greedy_rewards)
    
    action_proba = [0.0] * 4
    action_proba[action_id] = 1.0
    
    return action_proba

def compare_with_greedy(policy, env):
    """
    Simple regret policy.
    """
    _, policy_rewards, __ = execute_policy(policy, env)
    _, pseudooptimal_rewards, __ = execute_policy(policy_greedy, env)
    
    return np.sum(policy_rewards) - np.sum(pseudooptimal_rewards)