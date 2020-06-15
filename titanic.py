
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv('../Titanic Dataset/train.csv')
# print(train_data.head())

test_data = pd.read_csv('../Titanic Dataset/test.csv')
# print(test_data.head())

general_data = pd.read_csv('../Titanic Dataset/gender_submission.csv')
# print(general_data.head())

# print(train_data.info())
# print(test_data.info())

# print(train_data.describe())
# print(test_data.describe())

