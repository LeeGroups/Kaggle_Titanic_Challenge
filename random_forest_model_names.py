import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

# Import the training data
train_data = pd.read_csv("./Data/train.csv")
test_data = pd.read_csv("./Data/test.csv")

""" # Code for splitting the first name, last name and title from the training data
new1 = test_data.Name.str.split(',',n = 2, expand = True)
new2 = new1[1].str.split('.',n = 2, expand = True)
last_name = new1[0]
title = new2[0]
name = new2[1]

train_data['last_name'] = last_name
train_data['title'] = title
train_data['name'] = name
train_data.head()

train_data.to_csv('train_split_names.csv', index=False) """

# Features we care about
features = ["Pclass", "Sex", "SibSp", "Parch","Age"]

# Limit the training to the only parameters we care about
train_data = train_data[features + ["Survived"]]

# delete the rows of corresponding to passengers that do not have an Age entry
train_data.dropna(inplace=True)

""" X_train, X_test, y_train, y_test = train_test_split(train_data.drop(["Survived"], axis=1), train_data["Survived"], test_size=0.33, random_state=42)

X = pd.get_dummies(X_train[features])
X_test_dummy = pd.get_dummies(X_test[features]) """

# Split the training data set, so that 1/4 of them are left as validation
x_train, x_test, y_train, y_test = train_test_split(train_data.drop(["Survived"], axis=1), train_data["Survived"], test_size=0.25, random_state=42)

X_train_dummy = pd.get_dummies(x_train[features])
X_test_dummy = pd.get_dummies(x_test[features])


# Defines a function that trains a RandomForestClassifier model using parameters depth and n_estimator, and outputs the accuracy of the model at predicting x against y
def parametric_model(depth, n_estimators, x = X_test_dummy, y = y_test):
    model = RandomForestClassifier(n_estimators = n_estimators, max_depth=depth, random_state=1)
    model.fit(X_train_dummy, y_train)
    predictions = model.predict(x)
    return accuracy_score(predictions,y)


# Initialize the depths we want to test
depths = range(1,20)

accuracy_depth_test = [parametric_model(depth,100) for depth in depths]
accuracy_depth_train = [parametric_model(depth,100, X_train_dummy, y_train) for depth in depths]

# Plot the accuracy of the prediction of the validation (in blue) and training set (in green) with varying depths 
plt.plot(depths, accuracy_depth_test,color='blue')
plt.plot(depths, accuracy_depth_train,color='green')
plt.show()


# Initialize the n-estimates we want to test
n_estimate = range(70,200,10)

accuracy_est_test = [parametric_model(6,estimate) for estimate in n_estimate]
accuracy_est_train = [parametric_model(6,estimate, X_train_dummy, y_train) for estimate in n_estimate]

# Plot the accuracy of the prediction of the validation (in blue) and training set (in green) with varying n_estimates
plt.plot(n_estimate, accuracy_est_test,color='blue')
plt.plot(n_estimate,accuracy_est_train,color='green')
plt.show()

#print(accuracy_score(predictions,y_test))

#print(balanced_accuracy_score(predictions,y_test))

#print(balanced_accuracy_score(model.predict(X),y_train))

#print(accuracy_score(model.predict(X),y_train))

#model.score(X_test_dummy,y_test)