import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Data preprocessing
# Load the dataset
data = pd.read_csv('housing_data.csv')

# Handle missing values if any
data.dropna(inplace=True)

# Encode categorical variables if necessary

# Split the data into features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train decision tree and random forest models
# Train decision tree
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Train random forest
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)

# Step 3: Evaluate the models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

dt_mae, dt_mse, dt_r2 = evaluate_model(dt_regressor, X_test, y_test)
rf_mae, rf_mse, rf_r2 = evaluate_model(rf_regressor, X_test, y_test)

# Step 4: Compare the performance of both models
print("Decision Tree:")
print("Mean Absolute Error:", dt_mae)
print("Mean Squared Error:", dt_mse)
print("R-squared:", dt_r2)
print("\nRandom Forest:")
print("Mean Absolute Error:", rf_mae)
print("Mean Squared Error:", rf_mse)
print("R-squared:", rf_r2)

# Conclusion
if rf_r2 > dt_r2:
    print("\nRandom Forest model is preferred for analyzing the data.")
else:
    print("\nDecision Tree model is preferred for analyzing the data.")
