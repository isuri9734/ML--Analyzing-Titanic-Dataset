import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('../ML--Analyzing-Titanic-Dataset/train.csv')
# print(train_data.head(10))

test_data = pd.read_csv('../ML--Analyzing-Titanic-Dataset/test.csv')
# print(test_data.head(10))

# print(train_data.info())
# print(test_data.info())

# print(train_data.describe())
# print(test_data.describe())

graph1 = train_data.pivot_table(index="Sex",values="Survived")
# graph1.plot.bar()
# plt.show()

graph2 = train_data.pivot_table(index="Pclass",values="Survived")
# graph2.plot.bar()
# plt.show()

p_survive = train_data[train_data["Survived"] == 1]
p_notsuvive = train_data[train_data["Survived"] == 0]
# p_survive["Age"].plot.hist(alpha=0.5,color='red',bins=50)
# p_notsuvive["Age"].plot.hist(alpha=0.5,color='blue',bins=50)

# plt.legend(['Passenger Survived','Passenger Died'])
# plt.show()

train_data = train_data.drop([ 'Embarked', 'PassengerId',
                        'Name', 'SibSp', 'Parch', 'Cabin', 'Ticket'], axis=1)

# train_data=train_data.fillna(train_data.mean())
test_data=test_data.fillna(test_data.mean())

# train_data["Age"] = train_data["Age"].fillna(value=train_data["Age"].mean())
# test_data["Age"] = test_data["Age"].fillna(value=test_data["Age"].mean())
# train_data['Age'] = train_data['Age'].astype(np.int64)
# test_data['Age'] = test_data['Age'].astype(np.int64)
# train_data["Sex"] = train_data["Sex"] == 'male'
# test_data["Sex"] = test_data["Sex"] == 'male'

data = [train_data, test_data]

for full_data in data:
    full_data['Age'] = full_data['Age'].fillna(value=train_data["Age"].mean())
    full_data['Age'] = full_data['Age'].astype(np.int64)
    full_data['Sex'] = full_data['Sex'] == 'male'


target_value = train_data['Survived'].values
# print(target_value)

f_array = train_data[['Pclass', 'Sex', 'Age','Fare']].values   
# print(f_array)

fe_array = test_data[['Pclass', 'Sex', 'Age','Fare']].values

train_X, test_x, train_y, test_y = train_test_split(f_array,target_value, test_size=0.20,random_state=0)

model = LogisticRegression()

model.fit(train_X, train_y)

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(train_X, train_y)

model_train_file = "train_data.pkl"

tuple_data = (model, random_forest, train_X, train_y, test_x, test_y, f_array, target_value, fe_array)

with open(model_train_file, 'wb') as file:
    pickle.dump(tuple_data, file)

# pickle.dump(tuple_data, open("train_data.pkl", 'wb'))
