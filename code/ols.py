import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools.eval_measures import rmse

data = pd.read_csv('../data/cleaned_data.csv')
data = data.query("is_probabilistic == 1")
data_train = data.query("study == 'lugovskyy et al (2017)'")
data_test = data.query("study != 'lugovskyy et al (2017)'")

baseline_ols = smf.ols(f'a ~ delta_rd', data = data).fit()
baseline_ols_control = smf.ols(f'a ~ delta_rd + prev_avg_a', data = data).fit()

print(summary_col([baseline_ols, baseline_ols_control], stars = True))

ols_initial_first_supergame = smf.ols(f'a ~ delta_rd', data = data.query("period == 1 & sequence == 1")).fit()
ols_initial_non_first_supergame = smf.ols(f'a ~ delta_rd', data = data.query("period == 1 & sequence != 1")).fit()
ols_initial_non_first_supergame_control = smf.ols(f'a ~ delta_rd + prev_super_game_payoff', data = data.query("period == 1 & sequence != 1")).fit()
print(summary_col([ols_initial_first_supergame, ols_initial_non_first_supergame, ols_initial_non_first_supergame_control], stars = True))

baseline_ols = smf.ols(f'a ~ delta_rd', data = data+train).fit()

