# import modules
import numpy as np
import pandas as pd
import csv
import math
import networkx as nx
import random
import datetime
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.integrate import odeint
from matplotlib.colors import to_rgba
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import rc
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['font.sans-serif'] = "Arial"
rcParams['font.family'] = "sans-serif"
rcParams['hatch.linewidth'] = 3
plt.rc('text', usetex = True)
plt.rc('font', family = 'Times New Roman', weight = 'bold')
rcParams['mathtext.fontset'] = 'cm'
rcParams['text.latex.preamble'] = r"""\usepackage{bm}"""
rcParams['axes.facecolor'] = 'white'
rcParams['axes.edgecolor'] = 'black'
rcParams['axes.grid'] = False

_Data_PATH_ = './data/' # where to save the data
_Figure_PATH_ = './figures/' # where to save the figures

class SI_model():
    """
    a class to simulate the SI Model
    A: network adjacency matrix
    T: iteration time
    param: dictionary of the parameters
    """
    
    def __init__(self, A, T, param):
        """
        a class to simulate the SI Model
        A: network adjacency matrix
        T: iteration time
        param: dictionary of the parameters
        """
        # network
        self.A = A
        # parameters
        self.N = A.shape[0]
        self.T = T
        self.epsilon = param['epsilon']
        self.beta = param['beta']
        self.gamma = param['gamma']
        self.p, self.b, self.c, self.D, self.r, self.K = param['probability'], param['b'], param['c'], param['D'], param['r'], param['K']
        # fractions of S individuals
        self.S = [self.N - int(self.N*self.epsilon)]
        self.state_h = np.array([0]*self.S[0] + [1]*(self.N - self.S[0])).reshape((self.N, ))  # S: 0, I: 1
        np.random.shuffle(self.state_h)
        # fractions of nq individuals
        self.nq = [self.N]
        self.state_q = np.array([0] * self.nq[0]).reshape((self.N, ))  # nq: 0, q: 1
        # quarantine time
        self.q_time = np.zeros(self.N).reshape((self.N, ))  # quarantine time
        # fractions of four types of individuals
        self.S_nq = [sum((1-self.state_h) * (1-self.state_q))]  #S&nq
        self.S_q = [sum((1-self.state_h) * self.state_q)]  #S&q
        self.I_nq = [sum(self.state_h * (1-self.state_q))]  #I&nq
        self.I_q = [sum(self.state_h * self.state_q)]  #I&q
        return None
    
    def SI_focal(self, i):
        '''
        update the health state of individual i
        i: the focal individual
        '''
        # initialization, beta = beta * k_{i,nq}/k
        beta = self.beta * sum(self.A[i] * self.state_h * (1-self.state_q)) / sum(self.A[i])
        # update
        ## S: 0, I: 1
        if self.state_h[i] == 0 and random.random() < beta: 
            self.state_h[i] = 1
            self.S.append(self.S[-1] - 1)
        elif self.state_h[i] == 1 and random.random() < self.gamma:
            self.state_h[i] = 0
            self.S.append(self.S[-1] + 1)
        else:
            self.S.append(self.S[-1])
        return self.state_h, self.S
    
    # perfect rationality
    def perfect_rationality(self, B, C):
        if B - C > 0:
            f = 0
        else:
            f = 1
        return f

    # imperfect rationality
    ## Fermi function
    def imperfect_rationality(self, B, C, K):
        f = 1 / (1 + math.exp((B - C)/K))
        return f

    def qnq_focal(self, i, method = 'perfect_rationality'):
        '''
        update the quarantine state of individual i
        i: the focal individual
        method: perfect_rationality or imperfect_rationality
        '''
        # nq: 0, q: 1
        if self.q_time[i] == 0:     
            self.B = self.b * (sum(self.A[i] * (1-self.state_q)))    # B = b * D * k_{nq}, b = b * D
            if self.state_h[i] == 0:
                self.C = self.c * sum(self.A[i] * self.state_h * (1-self.state_q))  # C = c * k_{i, nq}
            else: 
                self.C = self.r * self.c * sum(self.A[i] * (1-self.state_h) * (1-self.state_q))  # C = r * c * k_{s, nq}
            if method == 'perfect_rationality':
                f = self.perfect_rationality(self.B, self.C)
                if f == 1:
                    self.state_q[i] = 1
                    self.nq.append(self.nq[-1] - 1)
                else:
                    self.nq.append(self.nq[-1])
            else:
                f = self.imperfect_rationality(self.B, self.C, self.K)
                if random.random() < f:
                    self.state_q[i] = 1
                    self.nq.append(self.nq[-1] - 1)
                else:
                    self.nq.append(self.nq[-1])
        else:
            self.nq.append(self.nq[-1])
        return self.state_q, self.nq
    
    def simulate(self, k):
        '''
        simulation
        k: the current iteration number in parallel computing
        '''
        # initialization
        self.S = [self.N - int(self.N*self.epsilon)]
        self.state_h = np.array([0]*self.S[0] + [1]*(self.N - self.S[0])).reshape((self.N, ))  # S: 0, I: 1
        np.random.shuffle(self.state_h)
        self.nq = [self.N]
        self.state_q = np.array([0] * self.nq[0]).reshape((self.N, ))  # nq: 0, q: 1
        self.q_time = np.zeros(self.N).reshape((self.N, ))  # quarantine time
        self.S_nq = [sum((1-self.state_h) * (1-self.state_q))]  #S&nq
        self.S_q = [sum((1-self.state_h) * self.state_q)]  #S&q
        self.I_nq = [sum(self.state_h * (1-self.state_q))]  #I&nq
        self.I_q = [sum(self.state_h * self.state_q)]  #I&q
        
        for d in range(self.T):
            # randomly pick an individual
            i = random.randint(0, self.N - 1)
            # update
            if random.random() < self.p:
                # update health state
                self.state_h, self.S= self.SI_focal(i)
                self.nq.append(self.nq[-1])
            else:
                # update social state
                self.state_q, self.nq = self.qnq_focal(i, method = 'perfect_rationality')
                self.S.append(self.S[-1])
            self.q_time += self.state_q  # update quarantine time
            for m in range(self.N):
                if self.q_time[m] == self.D:
                    self.state_q[m] = 0
                    self.nq[-1] += 1
                    self.q_time[m] = 0
            self.S_nq.append(sum((1-self.state_h) * (1-self.state_q)))  #S&nq
            self.S_q.append(sum((1-self.state_h) * self.state_q))  #S&q
            self.I_nq.append(sum(self.state_h * (1-self.state_q)))  #I&nq
            self.I_q.append(sum(self.state_h * self.state_q))  #I&q
        return np.array(self.S)/self.N, np.array(self.nq)/self.N, np.column_stack((self.S_nq, self.S_q, self.I_nq, self.I_q))/self.N, np.array(self.state_h), np.array(self.state_q)
        
    def simulateH(self, k):
        '''
        simulation with only a health state
        k: the current iteration number in parallel computing
        '''
        self.S = [self.N - int(self.N*self.epsilon)]
        self.state_h = np.array([0]*self.S[0] + [1]*(self.N - self.S[0])).reshape((self.N, ))  # S: 0, I: 1
        np.random.shuffle(self.state_h)
        
        for d in range(self.T):
            # randomly pick an individual
            i = random.randint(0, self.N - 1)
            # update
            if random.random() < self.p:
                # update health state
                self.state_h, self.S = self.SI_focal(i)
            else:
                self.S.append(self.S[-1])
        return np.array(self.S)/self.N  
        
        