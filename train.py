import numpy as np
import pandas as pd     #this to read the data 
from matplotlib import pyplot as plt    #plots 
import pickle


data =  pd.read_csv('C:\\Users\\fahad\\Desktop\\GIT\\model\\train.csv')


data = np.array(data)   #this is to make the data as an array to play with 
m, n = data.shape       #stores the rows and columns from the data 
np.random.shuffle(data)


#   dont want the model to memorize the data ,,, so we have a validation set of data 

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape

def init_params():
    W1 = np.random.rand(10, 784)-0.5
    b1 = np.random.rand(10, 1)-0.5
    W2 = np.random.rand(10, 10)-0.5
    b2 = np.random.rand(10, 1)-0.5
    return W1, b1, W2, b2

def load_weights(filename="model_weights.pkl"):
    import os
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            W1, b1, W2, b2 = pickle.load(f)
        print("Model weights loaded!")
        return W1, b1, W2, b2
    else:
        return init_params()  # If no saved model, start fresh

def save_weights(W1, b1, W2, b2, filename="model_weights.pkl"):
    with open(filename, "wb") as f:
        pickle.dump((W1, b1, W2, b2), f)
    print("Model weights saved!")


def ReLu(Z):
    return np.maximum(Z, 0)          # this is to make the function sochastic 

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))        # this is only to make the sum of probabilities to 1 ,,, and to optimize 
    return A

def forward_prop(W1,b1,W2,b2,X):
    Z1 = W1.dot(X) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1,A1,Z2,A2

def one_hot(Y):                 # in the end with softmax ,, to get which values are true if its a good guess or not 
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLu(Z):
    return Z > 0


def back_prop(Z1,A1,Z2,A2,W2 ,X,Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLu(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha *dW1 
    b1 = b1 - alpha *db1
    W2 = W2 - alpha *dW2
    b2 = b2 - alpha *db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradiant_descent(X, Y, iterations, alpha):
    # W1, b1, W2, b2 = init_params()
    W1, b1, W2, b2 = load_weights()
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1,b1,W2,b2,X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1,db1, dW2, db2, alpha)

        if i % 50 == 0:
            print("iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    save_weights(W1, b1, W2, b2)
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):              
    _, _, _, A2 = forward_prop(W1,b1,W2,b2,X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):              # for testing and prediction
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None],W1,b1,W2,b2 )
    label = Y_train[index ]
    print("prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255        # converting the image from data back to image with plot
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

  



W1, b1, W2, b2 = load_weights()
test_prediction(1,W1, b1, W2, b2)