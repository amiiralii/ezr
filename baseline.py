import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import random
import time
import csv
import os

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

def export(df_y, results, time, dataset, algo):
    try:
        os.mkdir(f'reg results/o{dataset.split('/')[-1][:-4]}/')
    except:
        pass

    with open(f'reg results/o{dataset.split('/')[-1][:-4]}/{algo}-{dataset.split('/')[-1]}', 'w', newline='') as f:    
        write = csv.writer(f)
        df_y.append('Time')
        write.writerow([y for y in df_y])
        for r in results:
            write.writerow(r)
        ll = (len(df_y)-1) * ['']
        ll.append(time)
        write.writerow(ll)

def calc_baseline(algo, dataset):
    st = time.time()
    results = []
    for _ in range(20):
        df = pd.read_csv(dataset)
        df_y = [c for c in df.columns if (c[-1] == '-' or c[-1] == '+')]
        X = df[[c for c in df.columns if (c[-1] != '-' and c[-1] != '+')]]
        res = []
        for target_column in df_y:
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(100*random.random()))

            if algo == 'linear':
                y_pred = linear( X_train, y_train, X_test)
            else:
                y_pred = lightgbm( X_train, y_train, X_test)

            res.append(round(mape(y_test, y_pred),3))
        results.append(res)

    export(df_y, results , round(time.time()-st,2), dataset, algo)


calc_baseline('linear', 'data/misc/Wine_quality.csv')
calc_baseline('lightgbm', 'data/misc/Wine_quality.csv')