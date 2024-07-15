import numpy as np
import random
import pandas as pd
from tqdm import tqdm

from player import Player

class PublicGoods:
    def __init__(self, delta, c, z, players):
        self.delta = delta
        self.c = c
        self.players = players
        self.n = len(players)
        self.z = z
        self.actions = np.linspace(0, 1, self.z)
        self.avg_a = None

    def sim_action(self, delta_rd, rd, s, k):
        actions = np.zeros(self.n)
        list_k = np.zeros(self.n)
        for i in range(self.n):
            self.players[i].choose_action(delta_rd, rd, self.avg_a)
        for i in range(self.n):
            player = self.players[i]
            u = 0
            range_k = int(player.ak)
            if range_k == 0:
                u = np.random.uniform(0, player.cutoffs[range_k])
            elif range_k == k - 1:
                u = np.random.uniform(player.cutoffs[range_k - 1], 1)
            else:
                u = np.random.uniform(player.cutoffs[range_k - 1], player.cutoffs[range_k])
            action = self.actions[(np.abs(self.actions - u)).argmin()]
            actions[i] = action
            if s not in player.history:
                player.history[s] = []
            player.history[s].append(action)
        self.avg_a = actions.mean()
        
    def get_reward(self, actions):
        joint_payoff = self.c / self.n * actions.sum()
        payoffs = np.zeros(self.n)
        for i, player in enumerate(self.players):
            payoffs[i] = joint_payoff - actions[i] + 1
            player.learn(payoffs[i])
        return payoffs
            #self.players[i].payoffs[player.init_k] += payoffs[i]

class Session:
    def __init__(self, delta, delta_rd, c, n, z, i, log=False):
        if i % n != 0:
            raise Exception(f"{i} is not divisible by {n}")
        self.delta = delta
        self.c = c
        self.n = n
        self.z = z
        self.i = i
        self.delta_rd = delta_rd
        self.players = [None]*i
        self.data = pd.DataFrame(columns = ['id', 'a', 'period', 'sequence', 'round', 'mpcr', 'c', 'n', 'delta_rd', 'delta', 'payoff']) 
        self.log = log

    def start_session(self, S, K):
        random.shuffle(self.players)
        games = []
        for i in range(0, self.i, self.n):
            games.append(PublicGoods(self.delta, self.c, self.z, self.players[i:i+self.n]))
        total_rd = 0
        for s in tqdm(range(S)):
            rd = 0
            avg_a = 0
            while True:
                for game in games:
                    game.sim_action(self.delta_rd, rd, s, K)
                for game in games:
                    actions = np.zeros(game.n)
                    for i, player in enumerate(game.players):
                        actions[i] = player.history[s][-1]
                    payoffs = game.get_reward(actions)
                    if self.log:
                        for i, player in enumerate(game.players):
                            newrow = {'id': id(player),
                                    'a': actions[i],
                                    'period': rd + 1,
                                    'sequence': s + 1,
                                    'round': total_rd + 1,
                                    'mpcr': self.c/self.n,
                                    'c': self.c,
                                    'n': self.n,
                                    'delta_rd': self.delta_rd,
                                    'delta': self.delta,
                                    'payoff': payoffs[i]}
                            self.data.loc[len(self.data)] = newrow
                            self.data = self.data.reset_index(drop=True)
                total_rd += 1  
                if np.random.uniform(0, 1) > self.delta:
                    break
                rd += 1

    def get_sim_data(self):
        return self.data
