import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data from the uploaded image (assuming it's in a CSV format)
data = pd.read_csv("/content/Real estate.csv")

# Drop the "No" column (assuming it's an index column)
data.drop(columns=["No"], inplace=True)

# Split the data into features (X) and target (Y)
X = data[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]
Y = data['Y house price of unit area']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the random forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
rf_model.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

#####################################################################################################################################################################################
DECISION TREE

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the data from the uploaded CSV file
data = pd.read_csv("Real estate.csv")

# Drop the "No" column (assuming it's an index column)
data.drop(columns=["No"], inplace=True)

# Split the data into features (X) and target (Y)
X = data.drop(columns=["Y house price of unit area"])
Y = data["Y house price of unit area"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the decision tree regressor
dt_model = DecisionTreeRegressor(random_state=42)

# Train the model on the training data
dt_model.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred_dt = dt_model.predict(X_test)

# Evaluate the model
mse_dt = mean_squared_error(Y_test, Y_pred_dt)
r2_dt = r2_score(Y_test, Y_pred_dt)

# Print the evaluation metrics for the decision tree model
print(f"Decision Tree - Mean Squared Error: {mse_dt:.2f}")
print(f"Decision Tree - R-squared: {r2_dt:.2f}")
