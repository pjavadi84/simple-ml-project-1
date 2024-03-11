import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# STEP 1: Load dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]




# STEP 2: Preparing Data:
#  A: Convert data to a pandas DataFrame
columns = [f"feature_{i}" for i in range(data.shape[1])]  # Creating column names for features
X = pd.DataFrame(data, columns=columns)
y = target
#  B: Split the data into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#STEP 3: Training the model by linear regression:unbiased estimate of how well your model is likely to perform on data it has never seen before.
# A: Initialize the model
model = LinearRegression()
# B: Fit the model
model.fit(X_train, y_train)



# STEP4: Evaluate the model and its performance on the test set
# A: Predictions
y_pred = model.predict(X_test)
# B: Calculate the mean squared error and R^2 Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



# STEP5(FINAL): Making prediction
# Predict the price of a sample house
sample_house = np.array([[0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.0900, 1, 296.0, 15.3, 396.90, 4.98]])
predicted_price = model.predict(sample_house)
print(f"Predicted Price: {predicted_price[0]}")