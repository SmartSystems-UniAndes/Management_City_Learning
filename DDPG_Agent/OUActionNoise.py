import tensorflow as tf
import numpy as np
# Ornstein-Uhlenbeck
# type of noise that models motion of particles in a lossless liquid with other particles moving at random 0.2 1 
class OUActionNoise(object): 
    def __init__(self, mu, sigma = 0, theta = 0, dt = 0.7e-2, x0 = None): #dt: differential with respect to time, noise correlated with t
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset() #sets previous value for the noise
       
    def __call__(self):
        x = self.x_prev+self.theta*(self.mu-self.x_prev)*self.dt+self.sigma+np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        
        self.xprev = x
        return x
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        
        