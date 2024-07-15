import os
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import logsumexp
import warnings

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS_DIR = os.path.join(ROOT_DIR, 'figs')

def entropy(n, z):
    return n*np.log(z+1)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    #return np.exp(np.log(x) - logsumexp(x))

def qbr(experiences):
    #e_x = np.exp(experiences)
    #return e_x/(1 + e_x.sum())
    return np.exp(experiences - logsumexp(np.insert(experiences, 0, 0))) 

def custom_round(number):
    return int(number + 0.5) if number > 0 else int(number - 0.5)

def round_to_nearest_half(number):
    return round(number * 2) / 2

def save_countour_plot(X, Y, Z, xlabel, ylabel, title, path, filename):
    directory = os.path.join(FIGS_DIR, path)
    os.makedirs(directory, exist_ok=True)
    plt.figure()
    contour = plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig_path = os.path.join(directory, filename)
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    plt.close()

def save_3d_plot(x, y, z, xlabel, ylabel, title, filename):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    scat = ax.scatter(x, y, z, c=z, cmap='viridis')
    plt.colorbar(scat)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.title(title)
    fig_path = os.path.join(FIGS_DIR, filename)
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    plt.close()

def save_2d_plot(x, y_list, xlabel, ylabel, title, path, filename, legend, ylim):
    directory = os.path.join(FIGS_DIR, path)
    os.makedirs(directory, exist_ok=True)
    fig = plt.figure()
    for y in y_list:
        plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(legend)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    fig_path = os.path.join(directory, filename)
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    plt.close()

def save_hist(plot, xlabel, ylabel, title, filename):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    fig_path = os.path.join(FIGS_DIR, filename)
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    plt.close()

def save_sim_fit(data, simulation_list, simulation_label_list, xlabel, ylabel, title, path, filename, ylim):
    assert len(simulation_list) == len(simulation_label_list)
    K = len(simulation_list)
    directory = os.path.join(FIGS_DIR, path)
    os.makedirs(directory, exist_ok=True)
    plt.hist(data, label = "data",  weights=np.ones(len(data)) / len(data))
    for k in range(K):
        simulation = simulation_list[k]
        label = simulation_label_list[k]
        counts, bins =np.histogram(simulation, bins = 10, weights = np.ones(len(simulation)) / len(simulation))
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        plt.plot(bin_centers, counts, label = label)
        #plt.plot(bin_centers, counts, label = "simulation")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    fig_path = os.path.join(directory, filename)
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    plt.close()

def save_sim_extrapolate(data, avg, low_percentile, top_percentile, xlabel, ylabel, title, path, filename, ylim, colors):
    assert len(avg) == len(low_percentile)
    assert len(avg) == len(top_percentile)
    assert len(avg) == len(colors)
    N = len(avg)
    directory = os.path.join(FIGS_DIR, path)
    os.makedirs(directory, exist_ok=True)
    for i in range(N):
        plt.plot(low_percentile[i], color=colors[i], linestyle=":")
        plt.plot(avg[i], color=colors[i], linestyle=":")
        plt.plot(top_percentile[i], color=colors[i], linestyle=":")
    plt.bar(np.arange(len(data)), data, color="black", alpha = 0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    fig_path = os.path.join(directory, filename)
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    plt.close()

def save_sim_hypothetical(avg, low_percentile, top_percentile, xlabel, ylabel, title, path, filename, ylim, colors):
    assert len(avg) == len(low_percentile)
    assert len(avg) == len(top_percentile)
    assert len(avg) == len(colors)
    N = len(avg)
    directory = os.path.join(FIGS_DIR, path)
    os.makedirs(directory, exist_ok=True)
    for i in range(N):
        plt.plot(low_percentile[i], color=colors[i], linestyle=":")
        plt.plot(avg[i], color=colors[i], linestyle=":")
        plt.plot(top_percentile[i], color=colors[i], linestyle=":")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale('log')
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    fig_path = os.path.join(directory, filename)
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    plt.close()

def save_eval_plots(model, bb, title, xlabel, ylabel, path, filename, logscale=False):
    directory = os.path.join(FIGS_DIR, path)
    os.makedirs(directory, exist_ok=True)
    fig, ax = plt.subplots()
    for column in model.columns:
        ax.plot(model.index, model[column], label=column)

    for column in bb.columns:
        ax.plot(bb.index, bb[column], label=column, linestyle='--')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    if logscale:
        plt.yscale('log')
    fig_path = os.path.join(directory, filename)
    plt.savefig(fig_path)
    print(f"Figure saved at {fig_path}")
    plt.close()

