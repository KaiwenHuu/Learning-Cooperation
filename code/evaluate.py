import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCURACY_DIR = os.path.join(ROOT_DIR, 'result', 'in_treat', 'accuracy')
LOSS_DIR = os.path.join(ROOT_DIR, 'result', 'in_treat', 'loss')
ACCURACY_CROSS_DIR = os.path.join(ROOT_DIR, 'result', 'cross_treat', 'accuracy')
LOSS_CROSS_DIR = os.path.join(ROOT_DIR, 'result', 'cross_treat', 'loss')

acc = pd.read_csv(os.path.join(ACCURACY_DIR, 'accuracy.csv'))
bb_acc = pd.read_csv(os.path.join(ACCURACY_DIR, 'blackbox_accuracy.csv'))
loss = pd.read_csv(os.path.join(LOSS_DIR, 'loss.csv'))
bb_loss = pd.read_csv(os.path.join(LOSS_DIR, 'blackbox_loss.csv'))

acc.set_index('k', inplace=True)
bb_acc.set_index('k', inplace=True)
loss.set_index('k', inplace=True)
bb_loss.set_index('k', inplace=True)

utils.save_eval_plots(acc, bb_acc, "Out of sample accuracy", "K", "Accuracy", "eval/in_treat", "in_treat_accuracy")
utils.save_eval_plots(loss, bb_loss, "Out of sample loss", "K", "Loss (Log Scaled)", "eval/in_treat", "in_treat_loss", True)

acc = pd.read_csv(os.path.join(ACCURACY_CROSS_DIR, 'accuracy.csv'))
bb_acc = pd.read_csv(os.path.join(ACCURACY_CROSS_DIR, 'blackbox_accuracy.csv'))
loss = pd.read_csv(os.path.join(LOSS_CROSS_DIR, 'loss.csv'))
bb_loss = pd.read_csv(os.path.join(LOSS_CROSS_DIR, 'blackbox_loss.csv'))

acc.set_index('k', inplace=True)
bb_acc.set_index('k', inplace=True)
loss.set_index('k', inplace=True)
bb_loss.set_index('k', inplace=True)

utils.save_eval_plots(acc, bb_acc, "Out of sample accuracy", "K", "Accuracy", "eval/cross_treat", "cross_treat_accuracy")
utils.save_eval_plots(loss, bb_loss, "Out of sample loss", "K", "Loss (Log Scaled)", "eval/cross_treat", "cross_treat_loss", True)

