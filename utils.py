import matplotlib.pyplot as plt
import numpy as np

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

        
def plot(states, rewards, action_taken=None, axes=None):
    """Plot the state/reward diagram after a policy is executed
    
    :param states the states to use
    :param rewards the rewards to use
    :param action_taken if provided, show the actions performed in the reward plot graph
    :param axes if provided, use the given axes rather than using custom defined axes.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
    states = np.array(states)
    for i in range(4):
        axes[0].plot(states[:,i], label=labels[i]);
        
    axes[0].set_title("Statistics of the epidemic")
    axes[0].set_xlabel('weeks since start of epidemic')
    axes[0].set_ylabel('State s(t)')
    axes[0].legend()
    axes[1].plot(rewards);
    axes[1].set_title('Reward')
    axes[1].set_xlabel('weeks since start of epidemic')
    axes[1].set_ylabel('reward r(t)')
    
    if action_taken:
        colors = ['r', 'g', 'b', 'k']
        for i in range(4):
            axes[1].vlines(np.where(np.array(action_taken) == i), ymin = -0.050, ymax=0.00, colors=colors[i], linestyle='dashed')

    print('total reward', np.sum(rewards))