
#step 1
import pandas as pd

iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path) # loading csv
print(home_data) # check if you could read csv properly 

#step 2
avg_lot_size = round(sum(home_data['LotArea']) / len(home_data['LotArea'])) # solve avg rounded to nearest integer
newest_home_age = 2024 - max(home_data['YrSold']) # current year - sold year

print(avg_lot_size)
print(newest_home_age)
