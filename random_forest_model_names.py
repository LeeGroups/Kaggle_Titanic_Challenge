import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from graphing_param import plot_param_changes

""" #Code for splitting the first name, last name and title from the training data
train_data = pd.read_csv("./Data/train.csv")
test_data = pd.read_csv("./Data/test.csv")
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

# Import the training data
train_data = pd.read_csv("./Data/train_split_names.csv")
test_data = pd.read_csv("./Data/test_split_names.csv")

# Features we care about
features = ["Pclass", "Sex", "SibSp", "Parch","Age","title","last_name"]

# Limit the training to the only parameters we care about
train_data = pd.get_dummies(train_data[features + ["Survived"]])

# delete the rows of corresponding to passengers that do not have an Age entry
train_data.dropna(inplace=True)

# Split the training data set, so that 1/4 of them are left as validation
x_train, x_test, y_train, y_test = train_test_split(train_data.drop(["Survived"], axis=1), train_data["Survived"], test_size=0.25, random_state=42)

# Defines a function that trains a RandomForestClassifier model using parameters depth and n_estimator, and outputs the accuracy of the model at predicting x against y
def parametric_model(param, x = x_test, y = y_test):
    depth = param[0]
    estimate = param[1]
    model = RandomForestClassifier(n_estimators = estimate, max_depth=depth, random_state=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x)
    return accuracy_score(predictions,y)

#plot_param_changes([[[depth,100] for depth in range(2,40)],[[20,estimate] for estimate in range(70,200,10)]], [range(2,40),range(70,200,10)], parametric_model, x_test, y_test, x_train,y_train)

def SGD_model(param, x, y):  
    eta0 = param[0]  
    model = SGDClassifier(eta0 = eta0,learning_rate="adaptive")
    model.fit(x_train, y_train)
    predictions = model.predict(x)
    return accuracy_score(predictions,y)

plot_param_changes([[[1/10 ** x ] for x in range(-2,10)]], [range(-2,10)], SGD_model, x_test, y_test, x_train,y_train)



# Initialize the depths we want to test
# depths = range(1,40)

# accuracy_depth_test = [parametric_model(depth,100) for depth in depths]
# accuracy_depth_train = [parametric_model(depth,100, x_train, y_train) for depth in depths]

# Plot the accuracy of the prediction of the validation (in blue) and training set (in green) with varying depths 
# figure, (axis0,axis1) = plt.subplots(ncols=2)
# figure.tight_layout(pad=4.0)

# axis0.plot(depths, accuracy_depth_test,color='blue')
# axis0.plot(depths, accuracy_depth_train,color='green')
# axis0.set_title("Accuracy as a function of depth")

# Initialize the n-estimates we want to test
# n_estimate = range(70,200,10)

# accuracy_est_test = [parametric_model(15,estimate) for estimate in n_estimate]
# accuracy_est_train = [parametric_model(15,estimate, x_train, y_train) for estimate in n_estimate]

# Plot the accuracy of the prediction of the validation (in blue) and training set (in green) with varying n_estimates
# axis1.plot(n_estimate, accuracy_est_test,color='blue')
# axis1.plot(n_estimate,accuracy_est_train,color='green')
# axis1.set_title("Accuracy as a function of n_estimate")

# print(parametric_model(20, 100, x = x_test, y = y_test))

# plt.show()