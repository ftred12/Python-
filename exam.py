#Visualization of data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
prest = pd.read_csv('/content/manufactures.csv')
prest.plot(x='YEAR', y='ENERGY', style='o')
plt.xlabel('YEAR')
plt.ylabel('ENERGY')
plt.show()

#Regression Line
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
prest = pd.read_csv('/content/manufactures.csv')
X = prest[['YEAR']] 
y = prest['ENERGY'] 
regressor = LinearRegression()
regressor.fit(X, y)
plt.scatter(X, y, label='Data Points')
plt.plot(X, regressor.predict(X), color='red', label='Regression Line')
plt.xlabel('YEAR')
plt.ylabel('ENERGY')
plt.legend()
plt.show()

#Predict new data
import pandas as pd
from sklearn.model_selection import train_test_split
prest = pd.read_csv('/content/manufactures.csv')
data = prest.loc[:, ['YEAR', 'TEMPERATURE']] 
x = pd.DataFrame(data['YEAR'])
y = pd.DataFrame(data['TEMPERATURE'])
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(X_train.shape) 
print(X_test.shape)
print(y_train.shape) 
print(y_test.shape)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X_test)
y_pred
y_test.head(10)

#Get the MAE,MSE AND RMSE
from sklearn import metrics
import numpy as np
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

#Predict with the new model
import joblib
import pickle
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y) 
joblib.dump(model, 'linear_regression_model.pkl')
with open('linear_regression_model.pkl', 'wb') as file:
  pickle.dump(model, file)

import joblib
import numpy as np
import pandas as pd
new_X = np.array([6, 7, 8, 9, 10]).reshape(-1, 1) 
model = joblib.load('linear_regression_model.pkl')
predictions = model.predict(new_X) 
new_data = pd.DataFrame({'X': new_X.flatten(), 'Predicted_Y': predictions.flatten()})
print(new_data)
