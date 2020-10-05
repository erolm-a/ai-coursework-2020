import gym
import numpy as np

from .core import Agent
from .siqr import BatchSIQR

class Epidemic(Agent):
	# wrapper environment around BatchSIQR
	
	def __init__(self):
		# action space is set of interventions on beta
		self.actions = np.array([1, .0175, 0.5, 0.65]) # beta coeffs
		#self.actions = np.linspace(0, 1, num=20) # beta coeffs
		self.action_space = gym.spaces.Discrete(self.actions.shape[0])
		self.beta = 0.373
		self.N = 6e8 # population size
		self.I0= 2e4# initial infections
		self.action_repeat = 7 # number of days between actions
		self.steps_total = int(365/self.action_repeat) # episode length (in days)
		self.steps = None
		
		self.env = BatchSIQR(beta='beta', N=self.N, epsilon=self.I0/self.N)
		self.observation_space = self.env.observation_space
		
	def reset(self):
		self.steps = 0
		return self.env.reset().reshape(-1)
		
	def step(self, action):
		# map action to beta
		assert(self.steps is not None) # step called before reset
		assert(action >= 0) # action out of bounds
		assert(action < self.actions.shape[0]) # action out of bounds
		
		c = self.actions[action]
		beta = self.beta * c
		r = 0
		for _ in range(self.action_repeat):
			s, _, d, i = self.env.step({'beta': beta})
			s = s.reshape(-1)
			r += self._reward(s/self.N, c)
		self.steps += 1
		
		# check done
		if self.steps >= self.steps_total:
			d = True
			self.steps = None
		
		return s, r/self.action_repeat, d, info
		
	def _reward(self, s, c):
		a = s[1] + s[2]
		# s: epidemic state (normalized)
		# c: policy severity
		b = 1-c
		return (-30*a - 30*a**2 - b - b**2)/62
		
		
		

    
