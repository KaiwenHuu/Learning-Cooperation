import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils

df_lugovsky = pd.read_csv('../data/tablereadyRev2.csv')
df_lugovsky['subject_treatment'] = list(zip(df_lugovsky['subject'], df_lugovsky['session'], df_lugovsky['treatment']))

df_mengel_peeters = pd.read_csv('../data/partner-hot.csv')
df_mengel_peeters = df_mengel_peeters.rename(columns={'Period': 'period', 'Group': 'group', 'Subject': 'id', 'Contr': 'a'})
df_mengel_peeters = df_mengel_peeters.dropna(subset=['id'])

df_mengel_peeters['id'] = df_mengel_peeters['id'].astype(int)
df_mengel_peeters['a'] = df_mengel_peeters['a']/10
df_mengel_peeters['n'] = 4
df_mengel_peeters['avg_a'] = df_mengel_peeters['TotalContr']/df_mengel_peeters['n']/10
df_mengel_peeters['c'] = 2
df_mengel_peeters['mpcr'] = df_mengel_peeters['c']/df_mengel_peeters['n']
df_mengel_peeters['delta'] = 0.9
df_mengel_peeters['round'] = df_mengel_peeters.groupby(['id']).cumcount() + 1
df_mengel_peeters['sequence'] = 1
df_mengel_peeters['binary'] = 0
df_mengel_peeters['session'] = 1
df_mengel_peeters['is_probabilistic'] = 1
df_mengel_peeters['round'] = df_mengel_peeters['round'].astype(int)
df_mengel_peeters = df_mengel_peeters.sort_values(by=['id', 'round'])
df_mengel_peeters = df_mengel_peeters.reset_index(drop = True)
df_mengel_peeters['prev_avg_a'] = df_mengel_peeters.groupby('id')['avg_a'].shift(1, fill_value=0)
df_mengel_peeters['prev_avg_a'] = np.where(df_mengel_peeters['period'] == 1, 0, df_mengel_peeters['prev_avg_a'])
df_mengel_peeters['payoff'] = 1 - df_mengel_peeters['a'] + df_mengel_peeters['mpcr'] * df_mengel_peeters['avg_a'] * df_mengel_peeters['n']
df_mengel_peeters['study'] = 'Mengel and Peeters (2011)'

df_lugovsky['prob'] = df_lugovsky['title'].str.extract(r'\d+\.\s*(.*?),\s*MPCR=[0-9\.]+')
df_lugovsky['is_probabilistic'] = np.where(df_lugovsky['prob'] == 'Probabilistic', 1, 0)
df_lugovsky['delta'] = 0.8
#df_lugovsky = df_lugovsky.drop(columns=['prob'])
#df_lugovsky = df_lugovsky[df_lugovsky.is_probabilistic == 1]

df_lugovsky['id'] = pd.factorize(df_lugovsky['subject_treatment'])[0].astype(int)
df_lugovsky = df_lugovsky.drop(columns=['subject_treatment'])

df_lugovsky = df_lugovsky.rename(columns={'contribution': 'a', 'groupfundtotal': 'avg_a', 'groupsize': 'n'})
df_lugovsky['a'] = df_lugovsky['a']/100
df_lugovsky['avg_a'] = df_lugovsky['avg_a']/100
df_lugovsky['mpcr'] = df_lugovsky['title'].str.extract(r'MPCR=([0-9\.]+)')
df_lugovsky['mpcr'] = df_lugovsky['mpcr'].astype(float)
df_lugovsky['c'] = df_lugovsky['mpcr']*df_lugovsky['n']

#df_lugovsky['prob'] = df_lugovsky['title'].str.extract(r'\d+\.\s*(.*?),\s*MPCR=[0-9\.]+')
#df_lugovsky['is_probabilistic'] = np.where(df_lugovsky['prob'] == 'Probabilistic', 1, 0)
#df_lugovsky['delta'] = np.where(df_lugovsky['is_probabilistic'] == 1, 0.8, 0)
#df_lugovsky = df_lugovsky.drop(columns=['prob'])
#df_lugovsky = df_lugovsky[df_lugovsky.is_probabilistic == 1]

#df_lugovsky['id'] = pd.factorize(df_lugovsky['subject_treatment'])[0].astype(int)
#df_lugovsky = df_lugovsky.drop(columns=['subject_treatment'])

df_lugovsky['round'] = df_lugovsky.groupby(['id']).cumcount() + 1
df_lugovsky['round'] = df_lugovsky['round'].astype(int)

df_lugovsky = df_lugovsky.sort_values(by=['id', 'round'])
df_lugovsky = df_lugovsky.reset_index(drop=True)

df_lugovsky['prev_avg_a'] = df_lugovsky.groupby('id')['avg_a'].shift(1, fill_value=0)
df_lugovsky['prev_avg_a'] = np.where(df_lugovsky['period'] == 1, 0, df_lugovsky['prev_avg_a'])
df_lugovsky['payoff'] = 1-df_lugovsky['a']+df_lugovsky['mpcr']*(df_lugovsky['avg_a']*df_lugovsky['n'])

df_lugovsky['study'] = 'lugovskyy et al (2017)'

print(f"num rows in lugovsky is {len(df_lugovsky)}")
print(f"num rows in mengel and peeters is {len(df_mengel_peeters)}")

data = pd.concat([df_lugovsky, df_mengel_peeters], axis=0, ignore_index=True, sort=False)

print(f"num rows in combined dataset is {len(data)}")

data['[0,0.25)'] = np.where(
        (data['period'] != 1) & (data['prev_avg_a'] < 0.25),
        1,
        0
    )

data['[0.25,0.5)'] = np.where(
        (data['period'] != 1) & (data['prev_avg_a'] >= 0.25) & (data['prev_avg_a'] < 0.50),
        1,
        0
    )

data['[0.5,0.75)'] = np.where(
        (data['period'] != 1) & (data['prev_avg_a'] >= 0.5) & (data['prev_avg_a'] < 0.75),
        1,
        0
    )

data['[0.75,1]'] = np.where(
        (data['period'] != 1) & (data['prev_avg_a'] >= 0.75),
        1,
        0
    )

data['[0,0.20)'] = np.where(
        (data['period'] != 1) & (data['prev_avg_a'] < 0.20),
        1,
        0
    )

data['[0.20,0.40)'] = np.where(
        (data['period'] != 1) & (data['prev_avg_a'] >= 0.2) & (data['prev_avg_a'] < 0.40),
        1,
        0
    )

data['[0.40,0.60)'] = np.where(
        (data['period'] != 1) & (data['prev_avg_a'] >= 0.4) & (data['prev_avg_a'] < 0.60),
        1,
        0
    )

data['[0.60,0.80)'] = np.where(
        (data['period'] != 1) & (data['prev_avg_a'] >= 0.6) & (data['prev_avg_a'] < 0.80),
        1,
        0
    )

data['[0.80,1]'] = np.where(
        (data['period'] != 1) & (data['prev_avg_a'] >= 0.8),
        1,
        0
    )

data['prev_cum_payoff'] = 0
for i, row in data.iterrows():
    if row['period'] != 1:
        data['prev_cum_payoff'][i] = data['payoff'][i-1] + data['prev_cum_payoff'][i-1]

data['prev_super_game_payoff'] = 0
data['prev_super_game_init'] = 0
data['prev_inter_diff_len'] = 0
for i, row in data.iterrows():
    if row['sequence'] == 1:
        curr_payoff = 0
        curr_init = 0
        curr_inter_diff_len = 0
    elif row['period'] == 1 and row['sequence'] != 1:
        if row['is_probabilistic'] == 1:
            curr_inter_diff_len = 1/(1-row['delta']) - data['period'][i-1]
        else:
            curr_inter_diff_len = 0
        curr_payoff = data['payoff'][i-1] + data['prev_cum_payoff'][i-1]
        nrounds = data['nrounds'][i-1]
        curr_init = data['a'][i-nrounds]
        data['prev_super_game_payoff'][i] = curr_payoff
        data['prev_super_game_init'][i] = curr_init
        data['prev_inter_diff_len'][i] = curr_inter_diff_len
    elif row['sequence'] != 1:
        data['prev_super_game_payoff'][i] = curr_payoff
        data['prev_super_game_init'][i] = curr_init
        data['prev_inter_diff_len'][i] = curr_inter_diff_len
data = data.drop(columns=['prev_cum_payoff'])

data['g'] = data['c']*(data['n']-1)/(data['n']*(data['c']-1))-1
data['l'] = -(data['c']-data['n'])/(data['n']*(data['c']-1))

data['delta_rd'] = round(data['delta'] - (data['g'] + data['l'])/(1 + data['g'] + data['l']), 3)
data['spe'] = round(data['g']/(1 + data['g']), 3)
data['rd'] = np.where(data['delta_rd'] >= 0, 1, 0)
data['spge'] = np.where(data['delta'] >= data['spe'], 1, 0)

#data['delta_rd'] = np.where(
#        data['delta'] == 0,
#        0,
#        (data['g']+data['l'])/(1+data['g']+data['l'])
#        )

data = data[data.binary == 0]

data['id'] = data.set_index(['id','study']).index.factorize()[0]+1

data = data.dropna(axis='columns', how='any')

data.to_csv('../data/cleaned_data.csv', index=False)

data_lugovskyy = data.query("study == 'lugovskyy et al (2017)' & is_probabilistic == 1")
data_mengel = data.query("study != 'lugovskyy et al (2017)'")
data_prob = data.query("is_probabilistic == 1")

print(f"number of observations in lugovskyy: {len(data_lugovskyy)}")
print(f"number of players in lugovskyy: {len(data_lugovskyy.id.unique())}")
print(f"number of observations in mengel and peters: {len(data_mengel)}")
print(f"number of players in mengel: {len(data_mengel.id.unique())}")


data_state_25 = data_prob[data_prob['[0,0.25)'] == 1]
utils.save_hist(data_state_25.hist(column='a', bins=5, range=(0,1)), 'action', 'frequency', 'Distribution of action when previous state was [0,0.25)', 'state1.png')

data_state_50 = data_prob[data_prob['[0.25,0.5)'] == 1]
utils.save_hist(data_state_50.hist(column='a', bins=5, range=(0,1)), 'action', 'frequency', 'Distribution of action when previous state was [0.25,0.5)', 'state2.png')

data_state_75 = data_prob[data_prob['[0.5,0.75)'] == 1]
utils.save_hist(data_state_75.hist(column='a', bins=5, range=(0,1)), 'action', 'frequency', 'Distribution of action when previous state was [0.5,0.75)', 'state3.png')

data_state_1 = data_prob[data_prob['[0.75,1]'] == 1]
utils.save_hist(data_state_1.hist(column='a', bins=5, range=(0,1)), 'action', 'frequency', 'Distribution of action when previous state was [0.75,1]', 'state4.png')

data_state_20 = data_prob[data_prob['[0,0.20)'] == 1]
utils.save_hist(data_state_20.hist(column='a', bins=5, range=(0,1)), 'action', 'frequency', 'Distribution of action when previous state was [0,0.20)', 'state5.png')

data_state_40 = data_prob[data_prob['[0.20,0.40)'] == 1]
utils.save_hist(data_state_40.hist(column='a', bins=5, range=(0,1)), 'action', 'frequency', 'Distribution of action when previous state was [0.20,0.40)', 'state6.png')

data_state_60 = data_prob[data_prob['[0.40,0.60)'] == 1]
utils.save_hist(data_state_60.hist(column='a', bins=5, range=(0,1)), 'action', 'frequency', 'Distribution of action when previous state was [0.40,0.60)', 'state7.png')

data_state_80 = data_prob[data_prob['[0.60,0.80)'] == 1]
utils.save_hist(data_state_80.hist(column='a', bins=5, range=(0,1)), 'action', 'frequency', 'Distribution of action when previous state was [0.60,0.80)', 'state8.png')

data_state_100 = data_prob[data_prob['[0.80,1]'] == 1]
utils.save_hist(data_state_100.hist(column='a', bins=5, range=(0,1)), 'action', 'frequency', 'Distribution of action when previous state was [0.80,1]', 'state9.png')
