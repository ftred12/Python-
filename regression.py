import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv('measure.csv')

X = df[['Speed', 'Vehicles', 'Rain', 'Fog', 'Drunkeness']]
y = df['Accident']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print('Coefficients: ', model.coef_)
 
print('Variance score: {}'.format(model.score(X_test, y_test)))
 
plt.style.use('fivethirtyeight')
 
plt.scatter(model.predict(X_train),
            model.predict(X_train) - y_train,
            color="green", s=10,
            label='Training')
 
plt.scatter(model.predict(X_test),
            model.predict(X_test) - y_test,
            color="blue", s=10,
            label='Testing')
 
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)
plt.legend(loc='lower right')
plt.title("Accidental severity")
plt.show()
