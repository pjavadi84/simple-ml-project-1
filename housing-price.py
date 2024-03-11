import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load dataset
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Preparing Data:
# STEP 1: Convert data to a pandas DataFrame
columns = [f"feature_{i}" for i in range(data.shape[1])]  # Creating column names for features
X = pd.DataFrame(data, columns=columns)

y = target

# STEP 2: Split the data into training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_test)


# Training the model by linear regression:unbiased estimate of how well your model is likely to perform on data it has never seen before.
# Initialize the model
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

print(model)