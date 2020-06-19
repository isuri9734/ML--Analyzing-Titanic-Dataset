import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pickle

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

# train_data["Age"] = train_data["Age"].fillna(value=train_data["Age"].mean())
# # print(train_data.info())

# test_data["Age"] = test_data["Age"].fillna(value=test_data["Age"].mean())
# # print(train_data.info())

p_survive = train_data[train_data["Survived"] == 1]
p_notsuvive = train_data[train_data["Survived"] == 0]
# p_survive["Age"].plot.hist(alpha=0.5,color='red',bins=50)
# p_notsuvive["Age"].plot.hist(alpha=0.5,color='blue',bins=50)

# plt.legend(['Passenger Survived','Passenger Died'])
# plt.show()

train_data=train_data.fillna(train_data.mean())
test_data=test_data.fillna(test_data.mean())

# train_data = train_data.dropna()
# test_data = test_data.dropna()

train_data = train_data.drop([ 'Embarked', 'PassengerId',
                        'Name', 'SibSp', 'Parch', 'Cabin'], axis=1)

train_data['Age'] = train_data['Age'].astype(np.int64)

test_data['Age'] = test_data['Age'].astype(np.int64)

train_data["Sex"] = train_data["Sex"] == 'male'
test_data["Sex"] = test_data["Sex"] == 'male'

# print(train_data.head(10))

target_value = train_data['Survived'].values
# print(target_value)

f_array = train_data[['Pclass', 'Sex', 'Age','Fare']].values   
# print(f_array)

fe_array = test_data[['Pclass', 'Sex', 'Age','Fare']].values

model = LogisticRegression()

X = f_array
y = target_value

train_X, test_x, train_y, test_y = train_test_split(X,y, test_size=0.20,random_state=0)

model1 = model.fit(train_X, train_y)

model2 = model.fit(X,y)

# model_train_file = "train_data.pkl"
# model_test_file = "test_data.pkl"

# with open(model_file, 'wb') as file:
#     pickle.dump(model1, file)

tuple_model1 = (model1, train_X, train_y, test_x, test_y)

tuple_model2 = ( model2, X, y, target_value, fe_array)


pickle.dump(tuple_model1, open("train_data.pkl", 'wb'))

pickle.dump(tuple_model2, open("test_data.pkl", 'wb'))
