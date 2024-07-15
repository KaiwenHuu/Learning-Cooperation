from abc import ABC, abstractmethod
import numpy as np
import random

import utils

STATES = [0.25, 0.5, 0.75]

class Player:
    def __init__(self, alpha, beta, lam, actions, cutoffs):
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.actions = actions
        self.cutoffs = cutoffs
        self.K = len(self.cutoffs) + 1
        self.payoffs = np.zeros(self.K)
        self.history = {}
    
    @abstractmethod
    def choose_action(self, delta_rd, rd, state):
        pass

    @abstractmethod
    def learn(self, payoff):
        pass

class IrlsgPlayer(Player):
    def __init__(self, alpha, beta, lam, sigma, actions, cutoffs):
        super().__init__(alpha, beta, lam, actions, cutoffs)
        self.sigma = sigma

    def choose_action(self, delta_rd, rd, state):
        if rd == 0:
            self.ak = self.choose_init_action(delta_rd)
        else:
            h = len(STATES)
            for s in range(len(STATES)):
                if state < STATES[s]:
                    h = s
                    break
            probs = self.sigma[h]
            probs = np.insert(probs, 0, 1-probs.sum())
            cdf = np.cumsum(probs)
            u = np.random.uniform(0, 1)
            for k in range(len(cdf)):
                if u <= cdf[k]:
                    self.ak = k
                    break

    def choose_init_action(self, delta_rd):
        utilities = self.alpha + delta_rd * self.beta
        for k in range(self.K - 1):
            utilities[k] += self.lam * (self.payoffs[self.K - 1] - self.payoffs[k])
        probs = utils.qbr(-utilities)
        probs = np.append(probs, 1-probs.sum())
        cdf = np.cumsum(probs)
        u = np.random.uniform(0, 1)
        for k in range(len(cdf)):
            if u <= cdf[k]:
                self.init_k = k
                break
        return self.init_k

    def learn(self, payoff):
        self.payoffs[self.init_k] += payoff

class RlPlayer(Player):
    def __init__(self, alpha, beta, lam, actions, cutoffs):
        super().__init__(alpha, beta, lam, actions, cutoffs)
    
    def choose_action(self, delta_rd, rd, state):
        utilities = self.alpha + delta_rd * self.beta
        for k in range(self.K - 1):
            utilities[k] += self.lam * (self.payoffs[self.K - 1] - self.payoffs[k])
        probs = utils.qbr(-utilities)
        probs = np.append(probs, 1-probs.sum())
        cdf = np.cumsum(probs)
        u = np.random.uniform(0, 1)
        for k in range(len(cdf)):
            if u <= cdf[k]:
                self.ak = k
                break

    def learn(self, payoff):
        self.payoffs[self.ak] += payoff
