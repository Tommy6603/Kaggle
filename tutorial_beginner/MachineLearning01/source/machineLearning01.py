
# Code you have previously used to load data
import pandas as pd

# Path of the file to read
iowa_file_path = '../data/train.csv'

home_data = pd.read_csv(iowa_file_path)

y = home_data['SalePrice']

# Create the list of features 
feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
X = home_data[feature_names]

# Review data X
print('X :')
print(X.head(3))

# create DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
iowa_model = DecisionTreeRegressor(max_depth=5, random_state=42)

# Fit the model
iowa_model.fit(X, y)

predictions = iowa_model.predict(X)
print(predictions)

# compare exact prices
print(y.head(5))
