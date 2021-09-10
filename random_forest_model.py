import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt

train_data = pd.read_csv("/Data/train.csv")
test_data = pd.read_csv("/Data/test.csv")
 

X_train, X_test, y_train, y_test = train_test_split(train_data.drop(["Survived"], axis=1), train_data["Survived"], test_size=0.33, random_state=42)

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(X_train[features])
X_test_dummy = pd.get_dummies(X_test[features])

#model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
#model.fit(X, y_train)
#predictions = model.predict(X_test_dummy)

def parametric_model(depth, n_estimators, x = X_test_dummy, y = y_test):
    model = RandomForestClassifier(n_estimators = n_estimators, max_depth=depth, random_state=1)
    model.fit(X, y_train)
    predictions = model.predict(x)
    return accuracy_score(predictions,y)

depths = range(1,10)
n_estimate = range(70,200,10)

accuracy_test = [parametric_model(6,estimate) for estimate in n_estimate]
accuracy_train = [parametric_model(6,estimate, X, y_train) for estimate in n_estimate]

plt.plot(n_estimate, accuracy_test,color='blue')
plt.plot(n_estimate,accuracy_train,color='green')
plt.show()

#print(accuracy_score(predictions,y_test))

#print(balanced_accuracy_score(predictions,y_test))

#print(balanced_accuracy_score(model.predict(X),y_train))

#print(accuracy_score(model.predict(X),y_train))

#model.score(X_test_dummy,y_test)