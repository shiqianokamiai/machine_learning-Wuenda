import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio.v2
from lr_utils import load_dataset
import pylab

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# train_set_x_orig train的图片矩阵
# train_set_y train的标签
print(train_set_y.shape)
print(train_set_x_orig.shape)  ## (209, 64, 64, 3)
# test_set_x_orig test的图片矩阵
# test_set_y test的标签
print(type(classes))

# Example of a picture
index = 5
plt.imshow(train_set_x_orig[index])  # show the fifth pic
pylab.show()
## csdn 博客plt.imshow 和 plt.show()的区别 https://blog.csdn.net/m0_66325043/article/details/131876441?ops_request_misc=&request_id=&biz_id=102&utm_term=plt.imshow%E5%92%8C%20plt.show%E7%9A%84%E5%8C%BA%E5%88%AB&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-131876441.142^v96^pc_search_result_base4&spm=1018.2226.3001.4187
## csdn 博客python 中plt.imshow(img)显示不了图片 https://blog.csdn.net/zqx951102/article/details/88861375?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170142564116800184117745%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170142564116800184117745&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-88861375-null-null.142^v96^pc_search_result_base4&utm_term=plt.imshow%E4%B8%8D%E6%98%BE%E7%A4%BA%E5%9B%BE%E7%89%87&spm=1018.2226.3001.4187
print("y = " + str(train_set_y[:, index]) + ", it is a " + classes[np.squeeze(train_set_y[:, index])].decode(
    "utf-8") + "'picture.")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print("Number of traning examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image:num_px = " + str(num_px))
print("Each image is of size:(" + str(num_px) + "," + str(num_px) + ", 3)")
print("train_set_x.shape:" + str(train_set_x_orig.shape))
print("train_set_y.shape:" + str(train_set_y.shape))
print("test_set_x.shape:" + str(test_set_x_orig.shape))
print("test_set_y.shape:" + str(test_set_y.shape))

#  reshape the training and test data sets so that images of size(num_px, num_px, 3)
#  are flattened into single vectors of shape (num_px*num_px*3, 1)
#  Flattening is a technique that is usesd to convert multi-dimensional arrays into a 1-D array.
#  It is generally used in Deep Learning when feeding the 1-D array information to the classification model.
#  The purpose of flattening is to decrease memory.

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))
# one common prepocessing step in machine learing is to center and standize your dataset, meaning that you substract the
# mean of the whole numpy array from each example.
# But for picture datasets. It is simpler and more convenient and work almost
# as well to just divided by 255.
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    :param z:  A scalar or numpy array of any size.
    :return: s -- sigmoid(z)
    """
    s = 1 + np.exp(-z)
    s = 1 / s
    return s


# initializing parameters
def initialize_with_zeros(dim):
    """
    The function creates a vector of zeros of shape(dim,1) for w and b
    :param dim: size of the w vector we want.
    :return:w -- initialized vector of shape(dim, 1)
            b -- initialized scalar
    """
    w = np.zeros([dim, 1])
    b = 0
    assert (w.shape == (dim, 1)) # 确认w的shape是否为(dim, 1)
    assert (isinstance(b, float) or isinstance(b, int)) # 确认b是小数或者整数。

    return w, b


# Forward and backward propagation
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above
    :param w: weights, a numpy array of size(num_px*num_px*3, 1)
    :param b: bias, a scalar
    :param X: data of size(num_px*num_px*3, number of examples)
    :param Y: true "label" vector
    :return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w
        db -- gradient of the loss with respect to b
    """
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = np.sum(np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T)) / (-m)
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    # csdn https://blog.csdn.net/fred_18/article/details/92688903 讲述了具体为什么要使用np.squeeze来处理机器学习的数据
    assert (cost.shape == ())
    grads = {"dw": dw,
             "db": db}

    return grads, cost


# optimization

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm.
    :param w: weights, a numpy array of size(num_px*num_px*3, 1)
    :param b: bias, a scalar
    :param X: data of shape(num_px*num_px*3, number of examples)
    :param Y: true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    :param num_iterations: number of iterations of the optimization loop
    :param learning_rate: learning rate of the gradient descent update rule
    :param print_cost: True to print the loss every 100 steps
    :return:
    """
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iterations %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}
    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

# 计算Yhat 将Yhat与activation相互比较，convert the entries of a into 0 (if activation <= 0.5)
# or 1 (if activation > 0.5) stores the predictions in a vector Y_prediction. If you wish, you can
# an if/else statement in a for loop(though there is also a way to vectorize this)
# prediction
def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression paramters (w, b)
    :param w: weights, a numpy array of size (num_px*num_px*3, 1)
    :param b: bias, a scalar
    :param X: data of size (num_px*num_px*3, number of examples)
    :return: Y_prediction: a numpy array(vector) containing all predictions (0/1) for the examples
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    print(A)
    print(A.shape)

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] += 1

    assert (Y_prediction.shape == (1, m))

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


d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
index = 10
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
print("y = " + str(test_set_y[0, index]) + ", you predicted that it is a \"" + classes[
    int(d["Y_prediction_test"][0, index])].decode("utf-8") + "\"picture.")

costs = np.squeeze(d['costs'])
# 画图
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations(per hunderds)')
plt.title("Learning rate = " + str(d["learning_rate"]))
plt.show()

learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is:" + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i,
                           print_cost=False)
    print('\n' + '----------------------------------------------' + "\n")

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel("cost")
plt.xlabel("iterations")

legend = plt.legend(loc="upper center", shadow=True)
frame = legend.get_frame()
frame.set_facecolor("0.90")
plt.show()

# 这里的代码与源代码有着严重的不一样，因为源代码中的scipy1.3.0开始，imread,imresize已经被弃用。
# 具体的原因在csdn https://blog.csdn.net/Mr_pandahu/article/details/122453774?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170151888916800180657883%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=170151888916800180657883&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-122453774-null-null.142^v96^pc_search_result_base4&utm_term=ndimage.imread&spm=1018.2226.3001.4187
# fname 放置自己图片的path来进行algorithm
fname = "D:/吴恩达深度学习作业/01.机器学习和神经网络/2.第二周 神经网络基础/编程作业/下载.jpg"
image = np.array(imageio.v2.imread(fname))
my_image = np.array(Image.fromarray(image).resize((num_px,num_px))).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
