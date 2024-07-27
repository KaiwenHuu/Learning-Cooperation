from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.neural_network import MLPClassifier
import numpy as np
import os
import pandas as pd

#TODO: add more columns to make it more high dimensional
COLUMNS = [
        'a',
        'sequence',
        'n',
        'c',
        'delta',
        'round',
        'prev_avg_a',
        '[0,0.25)',
        '[0.25,0.5)',
        '[0.5,0.75)',
        '[0.75,1]',
        'prev_super_game_payoff',
        'prev_super_game_init',
        'prev_inter_diff_len',
        'delta_rd',
        'spe',
        'rd',
        'spge'
        ]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(ROOT_DIR, 'result')

def get_acc_and_mse(X_train, y_train, X_test, y_test, K_range, acc_path, loss_path):
    acc_rows = []
    loss_rows = []
    for K in range(2, K_range+1):
        bins = [0]*(K-1)
        for k in range(K-1):
            bins[k] = (k+1)/K
        y_bin_train = np.digitize(y_train, bins)
        if K == 2:
            y_bin_validate = np.where(y_train >= 1/2, 1, 0)
            assert np.all(y_bin_validate == y_bin_train)
    
        y_bin_test = np.digitize(y_test, bins)
        nn = MLPClassifier(random_state=1, max_iter=10000).fit(X_train, y_bin_train)
        y_test_hat = nn.predict(X_test)
        y_test_hat_prob = nn.predict_proba(X_test)
        #nn_acc = np.mean(y_test_hat == y_bin_test)
        nn_acc = nn.score(X_test, y_bin_test)
        nn_loss = log_loss(y_bin_test, y_test_hat_prob)
        print(f"K = {K}: the nn accuracy is {nn_acc}")
        print(f"K = {K}: the nn loss is {nn_loss}")
    
        gbt_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, y_bin_train)
        gbt_acc = gbt_clf.score(X_test, y_bin_test)
        y_test_hat_prob = gbt_clf.predict_proba(X_test)
        gbt_loss = log_loss(y_bin_test, y_test_hat_prob)
        print(f"K = {K}: the gbt accuracy is {gbt_acc}")
        print(f"K = {K}: the gbt loss is {gbt_loss}")
    
        #ols = sm.OLS(y_bin_train, X_train[:,-1]).fit()
        ols = LinearRegression().fit(X_train[:,-1].reshape(-1, 1), y_bin_train)
        y_test_hat = ols.predict(X_test[:,-1].reshape(-1, 1))
        ols_loss = mean_squared_error(y_test_hat, y_bin_test)
        print(f"K = {K}: ols loss is {ols_loss}")

        lasso = LassoCV(cv=5, random_state=0).fit(X_train, y_bin_train)
        y_test_hat = lasso.predict(X_test)
        lasso_loss = mean_squared_error(y_test_hat, y_bin_test)
        print(f"K = {K}: lasso loss is {lasso_loss}")

        acc_row = pd.DataFrame(
                {'k': [K],
                 'mlp': [nn_acc],
                 #'mlp': [mlp_clf_acc],
                 'gbt': [gbt_acc]}
                 #'lasso': [lasso_acc]}
                )
        loss_row = pd.DataFrame(
                {'k': [K],
                 'mlp': [nn_loss],
                 #'mlp': [mlp_clf_mse],
                 'gbt': [gbt_loss],
                 'ols': [ols_loss],
                 'lasso': [lasso_loss]}
                )
        acc_rows.append(acc_row)
        loss_rows.append(loss_row)
    accuracies = pd.concat(acc_rows, ignore_index = True)
    accuracy_dir = os.path.join(RESULT_DIR, acc_path)
    os.makedirs(accuracy_dir, exist_ok=True)
    accuracy_path = os.path.join(accuracy_dir, "blackbox_accuracy.csv")
    accuracies.to_csv(accuracy_path, index = False)
    losses = pd.concat(loss_rows, ignore_index = True)
    loss_dir = os.path.join(RESULT_DIR, loss_path)
    os.makedirs(loss_dir, exist_ok=True)
    loss_path = os.path.join(loss_dir, "blackbox_loss.csv")
    losses.to_csv(loss_path, index = False)

data = pd.read_csv('../data/cleaned_data.csv')
data = data.query("is_probabilistic == 1")
data_lugovskyy = data.query("study == 'lugovskyy et al (2017)'")
data_lugovskyy_train = data_lugovskyy.query("session != 3")
data_lugovskyy_test = data_lugovskyy.query("session == 3")
data_lugovskyy_train = data_lugovskyy_train[COLUMNS]
data_lugovskyy_test = data_lugovskyy_test[COLUMNS]

y_train = data_lugovskyy_train['a'].to_numpy()
X_train = data_lugovskyy_train.drop(columns = ['a']).to_numpy()
y_test = data_lugovskyy_test['a'].to_numpy()
X_test = data_lugovskyy_test.drop(columns = ['a']).to_numpy()

get_acc_and_mse(X_train, y_train, X_test, y_test, 26, "in_treat/accuracy", "in_treat/loss")

data_train = data.query("study == 'lugovskyy et al (2017)'")
data_test = data.query("study != 'lugovskyy et al (2017)'")
data_train = data_train[COLUMNS]
data_test = data_test[COLUMNS]

y_train = data_train['a'].to_numpy()
X_train = data_train.drop(columns = ['a']).to_numpy()
y_test = data_test['a'].to_numpy()
X_test = data_test.drop(columns = ['a']).to_numpy()

get_acc_and_mse(X_train, y_train, X_test, y_test, 11, "cross_treat/accuracy", "cross_treat/loss") 
