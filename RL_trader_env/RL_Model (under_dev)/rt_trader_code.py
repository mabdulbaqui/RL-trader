from sklearn.model_selection import train_test_split
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gym_anytrading.envs import StocksEnv
from finta import TA
import os


def add_signals(env):
    start = env.frame_bound[0] - env.window_size
    end = env.frame_bound[1]
    prices = env.df.loc[:, 'Low'].to_numpy()[start:end]
    signal_features = env.df.loc[:, ['Low', 'Volume','SMA', 'RSI', 'OBV','MACD', 'MACD_signal']].to_numpy()[start:end]
    return prices, signal_features

class MyCustomEnv(StocksEnv):
    _process_data = add_signals


models_dir = "models/A2C"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)


df = pd.read_csv('all_stocks_5yr.csv')

data = df.drop(columns=["Date"])
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
outliers = df[~((df[1:] < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.drop(df[((df[1:] < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)].index ,inplace=True ,axis=0)
df.drop(['Name'],axis=1,inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df['SMA'] = TA.SMA(df, 12)
df['RSI'] = TA.RSI(df)
df['OBV'] = TA.OBV(df)
df[['MACD', 'MACD_signal']] = TA.MACD(df)
df.fillna(0, inplace=True)
df.set_index('Date',inplace=True)
x_train,x_test = train_test_split(df , test_size=0.1)
del df
env_train = MyCustomEnv(df=x_train, window_size=12, frame_bound=(12,x_train.shape[0]))

model = A2C('MlpPolicy' , env = env_train , verbose=1 ,tensorboard_log=logdir)
timesteps=10_000
for i in range(30):
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False , tb_log_name="A2C" )
    model.save(f"{models_dir} / {timesteps*i}")

