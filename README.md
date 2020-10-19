# ViRL

ViRL is an Epidemics Reinforcement Learning Environment. Agents are tasked with controlling the spread of a virus with one of four non-medical policy interventions: 

0. no intervention (remove all restrictions)
1. impose a full lockdown
2. implement track & trace
3. enforce social distancing and face masks

![image](img/ViRL_loop.png)

Once per week, the agent obtains evidence of the state of the epidemic and can update its policy accordingly. Each episode ends after 52 weeks, irrespective of the remaining number of infected individuals at that time.

Each intervention has a different impact on the infection rate, on the total number of simultaneously infected and hospitalized persons, and on the economic opportunity cost, which are summarized as a single scalar reward at each time step. 

The reward as a function of the infected population, and as a function of the action taken (at fixed level of infection) is illustrated in the figure below.

![image](img/reward.png)

## Disclaimer

ViRL is a toy environment for educational purposes on Reinforcement Learning algorithms. ViRL does not aim to replicate any epidemic in particular. The net effect of policy interventions is not grounded in reality and the net reward is entirely fictional. As a result, optimal policies on this toy example do not generalize to real-world epidemics.

## Getting Started

Ensure you have Python 3.5 or above, gym, numpy and matplotlib installed.

notebooks/random_agent.ipynb illustrates a basic (random) agent interacting with the environment.

## Author Contact

If you have any questions or feedback, please contact the author
[Sebastian Stein](mailto:sebastian.stein@glasgow.ac.uk)