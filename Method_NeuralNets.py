'''
This file contain all neutral Network related Functions
'''

import numpy as np

def sortEduKey(eduKeys, reverse=False):
    if reverse:
        eduKeys = [int(x) for x in eduKeys]
        eduKeys = (sorted(eduKeys, reverse=True))
        return eduKeys

'''
Activation Function Methods and derivates
'''
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - (np.tanh(x) ** 2)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigma(x):
    return np.divide((1.0), np.add((1.0), np.exp(np.negative(x))))

def sigmaprime(x):
    return np.multiply(sigmoid(x), np.subtract((1.0), sigmoid(x)))

def ReLU(x):
    return x * (x > 0)

def dReLU(x):
    return 1. * (x > 0)

'''
Weight initialization Function
'''
def initialize_weight_variable(d1, d2):
    initial = np.divide(np.random.rand(d1, d2).astype(np.float32), 100)
    return initial

'''
Feed forward Functions
'''
'''
if len((np.matmul(np.matrix(ab),W1)))== 1:
            return (np.tanh(np.matmul(np.matrix(ab),W1)).tolist())[0]
        return np.tanh(np.matmul(np.matrix(ab),W1))
'''
def feedforward_act(ab , W1, activationFunc):
    # if len(ab.shape())==1:
    #     ab=np.expand_dims(ab,0)
    #print ab

    if(activationFunc=="tanh"):
        #print W1.shape
        #print np.matrix(ab).shape
        if len((np.matmul(np.matrix(ab), W1))) == 1:
            return (np.tanh(np.matmul(np.matrix(ab), W1)).tolist())[0]
        return np.tanh(np.matmul(np.matrix(ab), W1))
    # elif (activationFunc == "sig"):
    #     return sigmoid(np.matmul(ab, W1))
    # elif (activationFunc == "relu"):
    #     return ReLU(np.matmul(ab, W1))
    # else:
    #     return (np.matmul(ab, W1))



def feedforward(ab , W1):
    return (np.matmul(np.matrix(ab), W1).tolist())[0]


'''
Error calculation Function
'''
def softmax_error(y, y_hat, y_in):
    return np.multiply(np.subtract( y, y_hat), tanh_deriv(y_in))

def non_softmax_error(delta, W2, input , W1):
    delta_in = np.matmul(np.matrix(delta), W2.T)
    return np.multiply(delta_in, tanh_deriv(feedforward(input, W1)))

'''
calculate delta
'''

def calculate_deltaW(error, input):
    delta_W = np.matmul(np.matrix(input).T,np.matrix(error))
    return delta_W

def BpthroughTree(EDUs, error_soft, W1, W2):
    delta_W_All = np.zeros([200, 100])
    eduKeys = sortEduKey(EDUs.keys(), reverse= True)
    parent_node = EDUs[str(eduKeys[0])]
    input = np.concatenate([EDUs[parent_node.leftChild].vector, EDUs[parent_node.rightChild].vector], 0)
    delta = non_softmax_error(error_soft, W2, input, W1)
    EDUs[parent_node.leftChild].delta = delta
    EDUs[parent_node.rightChild].delta = delta
    deltaw = calculate_deltaW(delta, input)
    delta_W_All = np.add(deltaw, delta_W_All)


    for key in eduKeys:
        if EDUs[str(key)].isLeaf == False and EDUs[str(key)].isRoot == False:
            input = np.concatenate([EDUs[EDUs[str(key)].leftChild].vector, EDUs[EDUs[str(key)].rightChild].vector], 0)

            W22= np.zeros([100, 100])
            if EDUs[str(key)].child == "left":
                W22 = W1[0:100, :]
            elif EDUs[str(key)].child == "right":
                W22 = W1[100:200, :]

            delta = non_softmax_error(EDUs[str(key)].delta, W22, input, W1)
            EDUs[EDUs[str(key)].leftChild].delta = delta
            EDUs[EDUs[str(key)].rightChild].delta = delta
            deltaw = calculate_deltaW(delta, input)
            delta_W_All = np.add(deltaw, delta_W_All)

    return delta_W_All

'''
Update NeuralNet Weight
'''

def update_weight(eta, W, delta_W):
    delta_W = np.multiply(eta, delta_W)
    return np.add(W, delta_W)