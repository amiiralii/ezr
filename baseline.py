import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random
import time

def mape(y_true, y_pred):
    return np.mean( np.abs(y_pred - y_true) / np.abs(y_true))

# Load the dataset
df = pd.read_csv('data/misc/Wine_quality.csv')
for i in ['Alcohol-', 'Quality+']:
    print(i,end=',\t')
print('time')

st = time.time()
for _ in range(20):
    df = pd.read_csv('data/misc/Wine_quality.csv')
    for target_column in ['Alcohol-','Quality+']:
        X = df.drop(columns=['Alcohol-', 'Quality+'])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(100*random.random()))

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate the model with MAPE
        print(f'{round(mape(y_test, y_pred),3)}', end='')
        if target_column != 'Quality+':
            print(',\t', end='')
    print()
print(',,,',round(time.time()-st,2))