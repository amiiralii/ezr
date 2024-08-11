import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import random
import time

def mape(y_true, y_pred):
    return np.mean( np.abs(y_pred - y_true) / np.abs(y_true))

def linear(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)

def lightgbm(X_train, y_train, X_test):
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
    'objective': 'regression',  # For regression tasks
    'metric': 'mape',           # Root Mean Squared Error
    'boosting_type': 'gbdt',    # Gradient Boosting Decision Tree
    'learning_rate': 0.1,       # Learning rate
    'num_leaves': 31,           # Number of leaves in one tree
    'verbose': -1               # Suppress warning messages
    }
    gbm = lgb.train(params, train_data, num_boost_round=100)
    return gbm.predict(X_test, num_iteration=gbm.best_iteration)


# Load the dataset
df = pd.read_csv('data/misc/Wine_quality.csv')
df_y = [c for c in df.columns if (c[-1] == '-' or c[-1] == '+')]

for i in df_y:
    print(i,end=',\t')
print('time')

st = time.time()
for _ in range(20):
    df = pd.read_csv('data/misc/Wine_quality.csv')
    df_y = [c for c in df.columns if (c[-1] == '-' or c[-1] == '+')]
    X = df[[c for c in df.columns if (c[-1] != '-' and c[-1] != '+')]]

    for target_column in df_y:
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(100*random.random()))

        #y_pred = linear( X_train, y_train, X_test)
        y_pred = lightgbm( X_train, y_train, X_test)

        # Evaluate the model with MAPE
        print(f'{round(mape(y_test, y_pred),3)}', end='')
        if target_column != df_y[-1]:
            print(',\t', end='')

    print()

print( (len(df_y)) * ',',round(time.time()-st,2))