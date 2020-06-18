# C:\ProgramData\Anaconda3\Script\activate tf_cpu

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score


train_data = pd.read_csv('../ML--Analyzing-Titanic-Dataset/train.csv')
# print(train_data.head())

test_data = pd.read_csv('../ML--Analyzing-Titanic-Dataset/test.csv')
# print(test_data.head())

# print(train_data.info())
# print(test_data.info())

# print(train_data.describe())
# print(test_data.describe())

train_data["Male/Female"] = train_data["Sex"] == 'male'
test_data["Male/Female"] = test_data["Sex"] == 'male'
# print(train_data.head())

plt.scatter(train_data['Fare'],train_data['Age'],c=train_data['Survived'],alpha=0.7)
plt.xlabel('Fare')
plt.ylabel('Age')
# plt.show()

plt.plot([32,110],[1.9,80])
# plt.show()

train_data = train_data.dropna()
test_data = test_data.dropna()

model = LogisticRegression()

f_array = train_data[['Fare','Age']].values
# print(f_array)

target_value = train_data['Survived'].values
# print(target_value)
# model.fit(f_array,target_value)

# print(model.coef_, model.intercept_)

fe_array = train_data[['Pclass', 'Male/Female', 'Age', 'SibSp', 'Parch', 'Fare']].values
# print(fe_array)

# model.fit(fe_array,target_value)

# print(model.predict([[1, False, 38.0, 1, 0, 71.2833]]))

# print(model.predict(fe_array[:5]))
# print(target_value[:5])

test_f_array = test_data[['Pclass', 'Male/Female', 'Age', 'SibSp', 'Parch', 'Fare']].values
# print(model.predict(test_f_array[:5]))
# print(target_value[:5])

test_prediction = model.predict(test_f_array)
# print((target_value == test_prediction).sum())

print (accuracy_score(target_value,test_prediction))

