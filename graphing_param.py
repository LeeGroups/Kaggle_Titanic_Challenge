import matplotlib.pyplot as plt

# a functions that takes in a model type, and plots the change the change in accuracy of the model with varying parameters
def plot_param_changes(param, axis, model_type, x_test, y_test, x_train,y_train):
    figure, plots = plt.subplots(ncols=len(param))
    if len(param) == 1:
        accuracy_test = [model_type(param[0][j],x_test,y_test) for j in range(len(param[0]))]
        accuracy_train = [model_type(param[0][j],x_train,y_train) for j in range(len(param[0]))]
        plots.plot(axis[0],accuracy_test,color='blue')
        plots.plot(axis[0],accuracy_train,color='green')
    else:
        figure.tight_layout(pad=4.0)
        for i in range(len(param)):
            accuracy_test = [model_type(param[i][j],x_test,y_test) for j in range(len(param[i]))]
            accuracy_train = [model_type(param[i][j],x_train,y_train) for j in range(len(param[i]))]
            plots[i].plot(axis[i],accuracy_test,color='blue')
            plots[i].plot(axis[i],accuracy_train,color='green')
    plt.show()

# Sample Input:

# plot_param_changes([[[depth,100] for depth in range(2,40)],[[20,estimate] for estimate in range(70,200,10)]], [range(2,40),range(70,200,10)], parametric_model, x_test, y_test, x_train,y_train)
# plot_param_changes([[[1/10 ** x ] for x in range(-2,10)]], [range(-2,10)], SGD_model, x_test, y_test, x_train,y_train)
