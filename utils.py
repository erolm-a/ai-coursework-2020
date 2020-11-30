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

    print('total reward', total_reward)
    

def evaluate(policy, full_eval=False, verbose=True, noisy=False):
    """
    Evaluate a policy
    
    :param policy a callable that, given in input a state, returns an action
    :param full_eval whether to fully evaluate the policy on all the problems or the first problem only
    :param verbose whether to get verbose output
    :param noisy whether to simulate a noisy environment
    """
    #trained_policy = create_policy(approximator_dl, 0, 4)
    limit = 10 if full_eval else 1
        
    envs = [virl.Epidemic(problem_id=i, noisy=noisy) for i in range(limit)]
    
    fig, axes = plt.subplots(limit, 2, figsize=(20, 8*limit))
    
    total_rewards = []
    
    for i, env in enumerate(envs):
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