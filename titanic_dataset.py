import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

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

train_data["Age"] = train_data["Age"].fillna(value=train_data["Age"].mean())
# print(train_data.info())

test_data["Age"] = test_data["Age"].fillna(value=test_data["Age"].mean())

p_survive = train_data[train_data["Survived"] == 1]
p_notsuvive = train_data[train_data["Survived"] == 0]
# p_survive["Age"].plot.hist(alpha=0.5,color='red',bins=50)
# p_notsuvive["Age"].plot.hist(alpha=0.5,color='blue',bins=50)

# plt.legend(['Passenger Survived','Passenger Died'])
# plt.show()

# gap = [0,5,10,15,20,40,60,120]
# train_data["age_cat"] = pd.cut(train_data["Age"], gap,labels=["Baby","Kids","Child","Teenager","Adult","M_adult","Aged"])
# test_data["age_cat"] = pd.cut(test_data["Age"], gap,labels=["Baby","Kids","Child","Teenager","Adult","M_adult","Aged"])

# new_graph = train_data.pivot_table(index="age_cat",values='Survived')

# # new_graph.plot.bar()
# # plt.show()

train_data = train_data.dropna()
test_data = test_data.dropna()

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

model.fit(train_X, train_y)

predict_value = model.predict(test_x)
# print(predict_value[:5])

# print(target_value[:5])

# accuracy1 = accuracy_score(test_y, predict_value)
# print(accuracy1)

predict_value2 = cross_val_score(model, X, y, cv=10)
predict_value2.sort()
# print(predict_value2)

accuracy2 = predict_value2.mean()
# print(accuracy2)

model.fit(X,y)
test_predict_value = model.predict(fe_array)

# print(test_prediction_value[:5])

# print(target_value[:5])

accuracy3 = test_predict_value.mean()
print(accuracy3)
