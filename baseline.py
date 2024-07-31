import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import random
import time
# Define SMAPE function
def mape(y_true, y_pred):
    return np.mean( np.abs(y_pred - y_true) / np.abs(y_true))

# Load the dataset
df = pd.read_csv('data/misc/auto93.csv')
for i in ['Lbs-', 'Acc+','Mpg+']:
    print(i,end=',\t')
print('time')
st = time.time()
for _ in range(20):
    df = pd.read_csv('data/misc/auto93.csv')
    for target_column in ['Lbs-', 'Acc+','Mpg+']:
        X = df.drop(columns=['Lbs-', 'Acc+','Mpg+', 'HpX'])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(100*random.random()))

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate the model with SMAPE
        print(f'{round(mape(y_test, y_pred),3)}', end='')
        if target_column != 'Mpg+':
            print(',\t', end='')
    print()
print(',,,',round(time.time()-st,2))