import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# train_set_x_orig train的图片矩阵
# train_set_y train的标签
print(train_set_y.shape)
print(train_set_x_orig.shape) ## (209, 64, 64, 3)
# test_set_x_orig test的图片矩阵
# test_set_y test的标签
print(type(classes))

# Example of a picture
index = 5
# plt.imshow(train_set_x_orig[index]) # show the fifth pic
print("y = " + str(train_set_y[:,index]) + ", it is a " + classes[np.squeeze(train_set_y[:,index])].decode("utf-8")+"'picture.")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print("Number of traning examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image:num_px = " + str(num_px))
print("Each image is of size:("+ str(num_px)+","+str(num_px)+ ", 3)")
print("train_set_x.shape:" + str(train_set_x_orig.shape))
print("train_set_y.shape:" + str(train_set_y.shape))
print("test_set_x.shape:" + str(test_set_x_orig.shape))
print("test_set_y.shape:" + str(test_set_y.shape))
##  reshape the training and test data sets so that images of size(num_px, num_px, 3) are flattened into single vectors of
## shape (num_px*num_px*3, 1)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
## one common prepocessing step in machine learing is to center and standize your dataset, meaning that you substract the
## mean of the whole numpy array from each example. But for picture datasets. It is simpler and more convenient and work almost
## as well to just divided by 255.
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

def sigmoid(z):
    s = 1 + np.exp(-z)
    s = 1/s
    return s
## initializing parameters
def initialize_with_zeros(dim):
    w = np.zeros([dim,1])
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b
## Forward and backward propagation
def propagate(w, b, X, Y):
     m = X.shape[1]
     A = sigmoid(np.dot(w.T, X) + b)
     cost = np.sum(np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1-A).T))/(-m)
     dw = np.dot(X, (A-Y).T)/m
     db = np.sum(A-Y)/m
     assert(dw.shape == w.shape)
     assert(db.dtype == float)
     cost = np.squeeze(cost)
     assert(cost.shape == ())
     grads = { "dw": dw,
               "db": db}

     return grads, cost
## optimization

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate*dw
        b = b - learning_rate*db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iterations %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


## prediction
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    print(A)
    print(A.shape)

    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_prediction[0,i] += 1

    assert(Y_prediction.shape == (1,m))

    return Y_prediction


# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    ### START CODE HERE ###

    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])
    # Gradient descent (≈ 1 line of code)
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)
    # Retrieve parameters w and b from dictionary "parameters"
    w = params["w"]
    b = params["b"]
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    ### END CODE HERE ###

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
index = 10
plt.imshow(test_set_x[:,index].reshape((num_px,num_px,3)))
print("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][0,index])].decode("utf-8") + "\"picture.")


costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations(per hunderds)')
plt.title("Learning rate = " + str(d["learning_rate"]))
plt.show()

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
        print("learning rate is:" + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
        print('\n' + '----------------------------------------------' + "\n")

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label = str(models[str(i)]["learning_rate"]))

plt.ylabel("cost")
plt.xlabel("iterations")

legend = plt.legend(loc = "upper center", shadow = True)
frame = legend.get_frame()
frame.set_facecolor("0.90")
plt.show()
