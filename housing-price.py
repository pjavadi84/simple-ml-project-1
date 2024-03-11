from sklearn.datasets import load_boston
import pandas as pd

# Load dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target