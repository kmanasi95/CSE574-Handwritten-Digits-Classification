import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1/(1 + np.exp(-1 * z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # set random seed
    np.random.seed(100)
    test_key_label = "test"
    train_key_label = "train"
    test_data = []
    test_label = []
    train_data = []
    train_label = []
    # load dataset
    for i in range(10):
        # load test data, test label
        test_key = test_key_label + str(i)
        test_val = mat[test_key]
        test_label.extend([i]*test_val.shape[0])
        test_data.extend(test_val)
    
        # load train data, train label
        train_key = train_key_label + str(i)
        train_val = mat[train_key]
        train_label.extend([i]*train_val.shape[0])
        train_data.extend(train_val)


    # convert list to np arrays
    test_data = np.asarray(test_data)
    test_label = np.asarray(test_label)
    
    train_data = np.asarray(train_data)
    train_label = np.asarray(train_label)
    
    
    # shuffle the array in the same order
    test_randomize = np.arange(len(test_label))
    np.random.shuffle(test_randomize)
    test_label = test_label[test_randomize]
    test_data = test_data[test_randomize]
    
    train_randomize = np.arange(len(train_label))
    np.random.shuffle(train_randomize)
    train_label = train_label[train_randomize]
    train_data = train_data[train_randomize]
    
    # divide training dataset into 5:1 set for validation
    partition = int((5/6)*len(train_label))
    validation_data = train_data[partition:]
    validation_label = train_label[partition:]
    train_data = train_data[:partition]
    train_label = train_label[:partition]
    # Feature selection
    # Your code here.
    
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    m = training_data.shape[0] # number of training data
    
    # one k hot encoding
    y = np.zeros((m, 10))
    for i in range(m):
        y[i,training_label[i]] = 1
        
    X = training_data # 50000 x 785
    
    # add bias in X (consider it as a1)
    #a1 = np.hstack((np.ones(m).reshape(m, 1), X)) # 50000 x 785 
    a1 = np.column_stack([X, np.ones((m,1),dtype = np.uint8)])
    
    # hidden layer propagation (use w1)
    z2 = np.dot(a1, w1.T) # 50000 x 50
    a2 = sigmoid(z2) # 50000 x 50
    
    # add bias in a2
    #a2 = np.hstack((np.ones(m).reshape(m, 1), a2)) # 50000 x 51
    a2 = np.column_stack([a2, np.ones((m,1),dtype = np.uint8)])
    
    # output layer propagation (use w2)
    z3 = np.dot(a2, w2.T) # 50000 x 10
    a3 = sigmoid(z3) # 50000 x 10
    
    # feed forward completed.
    
    t1 = w1[:,1:] # 50 x 784
    t2 = w2[:,1:] # 10 x 50
    
    #Start back propagation
    
    #Gradient Descent
    #Derivative of error function wrt weight from input feature to hidden unit
    dl = a3 - y
    dl_w = np.dot(dl,w2)
    z_prod = (1 - a2) * a2
    final_prod = dl_w * z_prod
    grad_w1 = np.dot(final_prod.T, a1)
    grad_w1 = grad_w1[0:n_hidden,:]
    #Derivative of error function wrt weight from hidden unit to output unit
    grad_w2 = np.dot(dl.T, a2)
    
    #End back propagation
    
    # regularization parameter
    reg = (lambdaval/(2*m))*np.asscalar(np.sum(np.square(t1)) + np.sum(np.square(t2)))
    # cost function for sigmoid function with regularization
    obj_val = (-1/m) * (np.sum(y*np.log(a3)) + np.sum((1-y)*np.log(1-a3))) + reg
    
    #New objective function wrt weight from input layer to hidden layer
    grad_w1 = (np.dot(lambdaval,w1) + grad_w1) / m
    #New objective function wrt weight from hidden layer to output layer
    grad_w2 = (np.dot(lambdaval,w2) + grad_w2) / m
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    #obj_grad = np.array([])

    return (obj_val, obj_grad)
    

def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    m = data.shape[0] # number of training data
    
    # add bias in X (consider it as a1)
    a1 = np.column_stack([data, np.ones((m,1),dtype = np.uint8)])
    
    # hidden layer propagation (use w1)
    z2 = np.dot(a1, w1.transpose()) # 50000 x 50
    a2 = sigmoid(z2) # 50000 x 50
    
    # add bias in a2
    a2 = np.column_stack([a2, np.ones((m,1), dtype = np.uint8)])
    
    # output layer propagation (use w2)
    z3 = np.dot(a2, w2.transpose()) # 50000 x 10
    a3 = sigmoid(z3) # 50000 x 10
    
    labels = np.argmax(a3, axis = 1)
    
    return labels

"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 20

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')