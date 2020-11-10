import matplotlib.pyplot as plt
import numpy as np

def plot(states, rewards, action_taken=None):
    """Plot the state/reward diagram after a policy is executed"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    labels = ['s[0]: susceptibles', 's[1]: infectious', 's[2]: quarantined', 's[3]: recovereds']
    states = np.array(states)
    for i in range(4):
        axes[0].plot(states[:,i], label=labels[i]);
        
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