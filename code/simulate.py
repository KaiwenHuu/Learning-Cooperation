import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import utils

from player import IrlsgPlayer, RlPlayer
from publicgoods import PublicGoods, Session

DELTA = 0.8
ESTIMATE = 'Estimate'
K = 10
I = 1000
N = 4
S = 15
Z = 26

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULT_DIR = os.path.join(ROOT_DIR, 'result')

RESULT_DIR_IRLSG = os.path.join(ROOT_DIR, 'data/result/irlsg_state_4_cross_section')
RESULT_DIR_RL = os.path.join(ROOT_DIR, 'data/result/reinforcement_learning_cross_section')

DF = pd.read_csv('../data/cleaned_data.csv')
DF_PROB = DF.query("is_probabilistic == 1")

def create_simulated_data(K, I, n, s, z, delta, c):
    #alpha1 = -0.2
    #beta1 = 0.5
    #lam1 = 0.2
    theta1 = np.array([-0.2, 0.5, 0.2])
    #alpha2 = -0.4
    #beta2 = 0.1
    #lam2 = 0.05
    theta2 = np.array([-0.4, 0.1, 0.05])
    pi = 1/5
    sigma = 1/1000

    actions = np.linspace(0, 1, z)
    cutoffs = (np.arange(1, K) / K)
    g = c*(n-1)/(n*(c-1))-1
    l = -(c-n)/(n*(c-1))
    delta_rd = delta - (g+l)/(1+g+l)

    samples1 = np.random.multivariate_normal(theta1, sigma*np.identity(3), int(I*pi))
    samples2 = np.random.multivariate_normal(theta2, sigma*np.identity(3), int(I*(1-pi)))

    samples = np.concatenate((samples1, samples2), axis = 0)
    session = Session(delta, delta_rd, c, n, z, I)
    for i, player in enumerate(session.players):
        params = samples[i]
        alpha = params[0:K-1]
        beta = params[K-1:2*K-2]
        lam = params[2*K-2]
        session.players[i] = RlPlayer(alpha = alpha, beta = beta, lam = lam, actions = actions, cutoffs = cutoffs)

    session.start_session(s, K)
    print(session.get_sim_data().head())
    sim_path = os.path.join(RESULT_DIR_RL, f"test_mixture_sim_C_{c}_N_{n}.csv")
    session.get_sim_data().to_csv(sim_path,index = False)
    return session.get_sim_data()

def sim_irlsg(K, I, n, s, z, delta, c, actions, cutoffs, delta_rd, path):
    # IRLSG model
    # Get mean and covariance of the mle estimates
    est_file_path = os.path.join(RESULT_DIR, path, "irlsg_state_4", f"K_{K}/est.csv")
    var_file_path = os.path.join(RESULT_DIR, path, "irlsg_state_4", f"K_{K}/var.csv")
    estimates = pd.read_csv(est_file_path)
    cov = pd.read_csv(var_file_path, header = None)
    
    mu = estimates[ESTIMATE].to_numpy()
    sigma = cov.to_numpy()

    # Draw I samples from the parameter for IRLSG
    samples = np.random.multivariate_normal(mu, sigma, I)

    session = Session(delta, delta_rd, c, n, z, I)
    for i, player in enumerate(session.players):
        params = samples[i]
        alpha = params[0:K-1]
        beta = params[K-1:2*K-2]
        lam = params[2*K-2]
        sigma = params[2*K-1:]
        session.players[i] = IrlsgPlayer(alpha = alpha, beta = beta, lam = lam, sigma = sigma.reshape(4, K-1), actions = actions, cutoffs = cutoffs)
    session.start_session(s, K)
    return session

def sim_rl(K, I, n, s, z, delta, c, actions, cutoffs, delta_rd, path):
    # RL model
    # Get mean and covariance of the mle estimates
    est_file_path = os.path.join(RESULT_DIR, path, "reinforcement_learning", f"K_{K}/est.csv")
    var_file_path = os.path.join(RESULT_DIR, path, "reinforcement_learning", f"K_{K}/var.csv")
    estimates = pd.read_csv(est_file_path)
    cov = pd.read_csv(var_file_path, header = None)
    
    mu = estimates[ESTIMATE].to_numpy()
    sigma = cov.to_numpy()

    # Draw I samples from the parameter for IRLSG
    samples = np.random.multivariate_normal(mu, sigma, I)

    session = Session(delta, delta_rd, c, n, z, I)
    for i, player in enumerate(session.players):
        params = samples[i]
        alpha = params[0:K-1]
        beta = params[K-1:2*K-2]
        lam = params[2*K-2]
        session.players[i] = RlPlayer(alpha = alpha, beta = beta, lam = lam, actions = actions, cutoffs = cutoffs)
    session.start_session(s, K)
    return session

def plot_fit(K, I, n, s, z, delta, c, path):
    actions = np.linspace(0, 1, z)
    cutoffs = (np.arange(1, K) / K)
    print(f"cutoffs are {cutoffs}")
    print(f"Z: {z}")
    g = c*(n-1)/(n*(c-1))-1
    l = -(c-n)/(n*(c-1))
    delta_rd = delta - (g+l)/(1+g+l)

    df_train_treatment = DF_PROB.query(f"delta == {delta} & c == {c} & n == {n}")
    session_irlsg = sim_irlsg(K, I, n, s, z, delta, c, actions, cutoffs, delta_rd, path)
    all_actions_irlsg = []
    for player in session_irlsg.players:
        all_actions_irlsg = all_actions_irlsg + sum(player.history.values(), [])
    all_actions_irlsg = np.array(all_actions_irlsg)    
    session_rl = sim_rl(K, I, n, s, z, delta, c, actions, cutoffs, delta_rd, path)
    all_actions_rl = []
    for player in session_rl.players:
        all_actions_rl = all_actions_rl + sum(player.history.values(), [])

    all_actions_rl = np.array(all_actions_rl)

    utils.save_sim_fit(df_train_treatment['a'], [all_actions_irlsg, all_actions_rl], ['irlsg_4', 'reinforcement_learning'], 'actions', 'percentage', f"C = {c}, N = {n}", "simulation", f"C_{c}_N_{n}_train_fit.png", [0, 0.75])


def init_plays(K, I, n, S, z, delta, c, path):
    actions = np.linspace(0, 1, z)
    cutoffs = (np.arange(1, K) / K)
    g = c*(n-1)/(n*(c-1))-1
    l = -(c-n)/(n*(c-1))
    delta_rd = delta - (g+l)/(1+g+l)
    
    session_irlsg = sim_irlsg(K, I, n, S, z, delta, c, actions, cutoffs, delta_rd, path) 
    init_actions_irlsg = np.zeros((S, I))

    for i, player in enumerate(session_irlsg.players):
        for s in range(S):
            init_actions_irlsg[s][i] = player.history[s][0]
    
    mean_irlsg = np.mean(init_actions_irlsg, axis = 1)
    std_irlsg = np.std(init_actions_irlsg, axis = 1)

    session_rl = sim_rl(K, I, n, S, z, delta, c, actions, cutoffs, delta_rd, path)
    init_actions_rl = np.zeros((S, I))

    for i, player in enumerate(session_rl.players):
        for s in range(S):
            init_actions_rl[s][i] = player.history[s][0]

    mean_rl = np.mean(init_actions_rl, axis = 1)
    std_rl = np.std(init_actions_rl, axis = 1)

    return delta_rd, [mean_irlsg, mean_rl], [mean_irlsg - std_irlsg, mean_rl - std_rl], [mean_irlsg + std_irlsg, mean_rl + std_rl]

def plot_init_actions(K, I, n, S, z, delta, c, path):
    delta_rd, avg, low_percentile, top_percentile = init_plays(K, I, n, S, z, delta, c, path)
    df_treatment = DF_PROB.query(f"delta == {delta} & c == {c} & n == {n}")
    df_treatment = df_treatment[df_treatment['period'] == 1]
    average_a = df_treatment.groupby('sequence')['a'].mean().reset_index()
    average_a.columns = ['sequence', 'average_a']

    utils.save_sim_extrapolate(average_a.average_a, avg, low_percentile, top_percentile, "super game", "average contribution rate", f"$\delta$ = {delta}, $\Delta^{{RD}}$ = {round(delta_rd, 3)}", "extrapolate", f"C_{c}_N_{n}_Z_{z}_extrapolate.png", [0,1], ["red", "blue"]) 

def plot_hypothetical_init_actions(K, I, n, S, z, delta, c, path):
    delta_rd, avg, low_percentile, top_percentile = init_plays(K, I, n, S, z, delta, c, path)
    utils.save_sim_hypothetical(avg, low_percentile, top_percentile, "super game (log scale)", "average contribution rate", f"$\delta$ = {delta}, $\Delta^{{RD}}$ = {round(delta_rd, 3)}", "hypothetical", f"C_{c}_N_{n}_Z_{z}_delta_{delta}.png", [0,1], ["red", "blue"]) 

def hypothetical_init_actions_xyz(K, I, n_list, S, z, delta_list, c_list, path):
    delta_rd_list = []
    mpcr_list = []
    lr_init_a_irlsg_list = []
    lr_init_a_rl_list = []
    n = len(n_list)
    assert n == len(delta_list)
    assert n == len(c_list)
    for i in range(n):
        delta_rd, avg, _, _ = init_plays(K, I, n_list[i], S, z, delta_list[i], c_list[i], path)
        delta_rd_list.append(delta_rd)
        mpcr_list.append(c_list[i]/n_list[i])
        lr_init_a_irlsg_list.append(avg[0][-1])
        lr_init_a_rl_list.append(avg[1][-1])
    utils.save_3d_plot(delta_rd_list, mpcr_list, lr_init_a_irlsg_list, "IRL-SG Long Run Cooperation Rate", "hypothetical_contour", "irlsg_contour.png")
    utils.save_3d_plot(delta_rd_list, mpcr_list, lr_init_a_rl_list, "RL Long Run Cooperation Rate", "hypothetical_contour", "rl_contour.png")

#df1 = create_simulated_data(2, I, 2, S, Z, DELTA, 1.2)
#df2 = create_simulated_data(2, I, 4, S, Z, DELTA, 2.4)
#df3 = create_simulated_data(2, I, 4, S, Z, DELTA, 1.2)
#
#sim_res = pd.concat([df1, df2, df3])
#sim_res['id'] = pd.factorize(list(zip(sim_res['id'], sim_res['delta_rd'])))[0]
#print(sim_res.head())
#sim_path = os.path.join(RESULT_DIR_RL, f"test_mixture_sim.csv")
#sim_res.to_csv(sim_path,index = False)
#plot_fit(K, I, 2, S, Z, DELTA, 1.2, "in_treat")
#plot_fit(K, I, 4, S, Z, DELTA, 2.4, "in_treat")
#plot_fit(K, I, 4, S, Z, DELTA, 1.2, "in_treat")
#
#plot_init_actions(K, I, 2, 50, Z, DELTA, 1.2, "all_prob")
#plot_init_actions(K, I, 4, 50, Z, DELTA, 2.4, "all_prob")
#plot_init_actions(K, I, 4, 50, Z, DELTA, 1.2, "all_prob")
#plot_init_actions(K, I, 4, 50, 11, 0.9, 2, "all_prob")
#
plot_hypothetical_init_actions(10, I*2, 4, 10000, 11, 0.5, 3, "all_prob")
plot_hypothetical_init_actions(10, I*2, 4, 10000, 11, 0.9, 3, "all_prob")
plot_hypothetical_init_actions(10, I*2, 4, 10000, 11, 0.9, 1.1, "all_prob")
plot_hypothetical_init_actions(10, I*2, 4, 10000, 11, 0.5, 1.1, "all_prob")
#plot_hypothetical_init_actions(10, I*10, 4, 10000, 11, 0.9, 3.9, "all_prob")
#plot_hypothetical_init_actions(10, I*10, 4, 10000, 11, 0.5, 3.9, "all_prob")
