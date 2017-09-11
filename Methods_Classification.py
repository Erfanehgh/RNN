'''
This file contains all classification Functions
'''
import os
from Method_NeuralNets import feedforward, tanh, softmax_error, calculate_deltaW, BpthroughTree, update_weight
from Method_RSTTree import readTree_att_Scalar, sortEduKey

def train_for_each_Sample (EDUs, y, W1, W2, eta):
    eduKeys = sortEduKey(EDUs.keys(), reverse=True)
    input2 = EDUs[str(eduKeys[0])].vector
    y_in = feedforward(input2, W2)
    output = tanh(y_in)
    error_soft = softmax_error(y, output, y_in)
    delta_W2 = calculate_deltaW(error_soft, input2)
    delta_W1 = BpthroughTree(EDUs, error_soft, W1, W2)
    W2 = update_weight(eta, W2, delta_W2)
    W1 = update_weight(eta, W1, delta_W1)

    return W1, W2

def train (path_Folder, WV, dim,  W1 , W2, eta, OutputFile, attScalar):

    posFileList = os.listdir(path_Folder + "pos/")
    negFileList = os.listdir(path_Folder + "neg/")
    numberofSamples = min(len(posFileList), len(negFileList))
    #numberofSamples=5
    for j in range(0, numberofSamples):

        path_File = path_Folder + "pos/" + posFileList[j]
        EDUs = readTree_att_Scalar(path_File, W1, WV, dim, "Satellite", attScalar, "tanh")
        if (len(EDUs) > 0):
            y = [1.0, 0]
            W1, W2 = train_for_each_Sample(EDUs, y, W1, W2, eta)

        path_File = path_Folder + "neg/" + negFileList[j]
        EDUs = readTree_att_Scalar(path_File, W1, WV, dim, "Satellite", attScalar, "tanh")

        if (len(EDUs) > 0):
            y = [0, 1.0]
            W1, W2 = train_for_each_Sample(EDUs, y, W1, W2, eta)

    return W1, W2


def test(path_Folder, mode, WV, dim,  W1 , W2, OutputFile, attScalar, iteration):
    posFileList_test = os.listdir(path_Folder + "pos/")
    negFileList_test = os.listdir(path_Folder + "neg/")
    numberofSamples_test = min(len(posFileList_test), len(negFileList_test))

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for k in range(0, numberofSamples_test):
        path_File_test = path_Folder + "pos/" + posFileList_test[k]
        EDUs = readTree_att_Scalar(path_File_test, W1, WV, dim, "Satellite", attScalar, "tanh")
        y = [1.0, 0]
        eduKeys = sortEduKey(EDUs.keys(), reverse=True)
        input2 = EDUs[str(eduKeys[0])].vector
        y_in = feedforward(input2, W2)
        output = tanh(y_in)
        #print output
        if output[0] >= output[1]:
            tp += 1
        else:
            fn += 1

        path_File_test = path_Folder + "neg/" + negFileList_test[k]
        EDUs = readTree_att_Scalar(path_File_test, W1, WV, dim, "Satellite", attScalar, "tanh")
        eduKeys = sortEduKey(EDUs.keys(), reverse=True)
        input2 = EDUs[str(eduKeys[0])].vector
        y_in = feedforward(input2, W2)
        output = tanh(y_in)

        if output[0] >= output[1]:
            tn += 1
        else:
            fp += 1

    accuracy = float(tp + fp) / (tp + tn + fp + fn)
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    F1 = 2 * (float(precision * recall)) / (precision + recall)

    print  attScalar, " ", iteration, " ", mode , " ", tp, " ", tn, " ", fp, " ", fn, " ", accuracy, " ", precision, " ", recall, " ", F1
    OutputFile.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (attScalar,  iteration, mode, tp, tn, fp, fn, accuracy, precision, recall, F1))