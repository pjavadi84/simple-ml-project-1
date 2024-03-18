import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# You can place this in a separate script or function that you only run once
def train_model():
    # Load dataset
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # Prepare Data
    columns = [f"feature_{i}" for i in range(data.shape[1])]
    X = pd.DataFrame(data, columns=columns)
    y = target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model evaluation - MSE: {mse}, R^2: {r2}")

    # Save the model to a file
    joblib.dump(model, 'linear_regression_model.joblib')

# You would call this function when you want to make a prediction
def predict_price(sample_house):
    # Load the trained model from the file
    model = joblib.load('linear_regression_model.joblib')

    # Predict the price of a sample house
    predicted_price = model.predict(sample_house)
    return predicted_price

# Usage example (this would not be in models.py, but wherever you are making the prediction):
# sample_house = np.array([[0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.0900, 1, 296.0, 15.3, 396.90, 4.98]])
# price = predict_price(sample_house)
# print(price)
