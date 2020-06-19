import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# with open(model_file, 'rb') as file:
#     Pickled_model = pickle.load(file)

# print(Pickled_model)

pickled_model, pickled_Xtrain, pickled_Ytrain, pickled_Xtest, pickled_Ytest = pickle.load(open("train_data.pkl", 'rb'))

pickled_model2,allx, ally, target_val, fearray = pickle.load(open("test_data.pkl", 'rb'))

predict_value = pickled_model.predict(pickled_Xtest)
# print(predict_value[:5])

# print(target_val[:5])

# accuracy1 = accuracy_score(pickled_Ytest, predict_value)
# print(accuracy1)



# predict_value2 = cross_val_score(pickled_model2, allx, ally, cv=5)
# print(predict_value2)

# accuracy2 = predict_value2.mean()
# print(accuracy2)


test_predict_value = pickled_model2.predict(fearray)

print(test_predict_value[:5])

print(target_val[:5])