import numpy as np
from Method_WV import readWV
from Method_NeuralNets import initialize_weight_variable
from Methods_Classification import train_AttWeight, test_AttWeight, train, test, calculateError_validation, test_AttWeight_DrHarati
import os
import math


'''

'''
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

WV=readWV("/home/erfaneh/Shared/WordVector/glove.6B/glove.6B.100d.txt", stop)
# path_Folder="/home/erfaneh/PycharmProjects/RNN/Data_Test/"
# path_Folder_test = "/home/erfaneh/PycharmProjects/RNN/Data/"
# path_Folder_val = "/home/erfaneh/PycharmProjects/RNN/Data/"

path_Folder="/home/erfaneh/Shared/IMDB-Parsed/Train/"
path_Folder_test = "/home/erfaneh/Shared/IMDB-Parsed/Test/"
path_Folder_val = "/home/erfaneh/Shared/IMDB-Parsed/Evaluation/"

eta = 0.001
dim = 100
activationFunc = "tanh"
OutputFile = open( "output_allData_AttNS.csv", "w")
OutputFile.write("iteration, data, tp, tn, fp, fn, accuracy, precision, recall, F-1 measure\n")

OutputFile_val_err = open( "val_Err_AttNS.csv", "w")
OutputFile_val_err.write("iteration, data, error\n")


# attScalar
'''

OutputFile.write("attScalar, iteration, data, tp, tn, fp, fn, accuracy, precision, recall, F-1 measure\n")
attOn = "Nucleus"
for attScalar in np.arange(1.0, 2.0, 0.1):

    W1 = initialize_weight_variable(200, 100)
    W2 = initialize_weight_variable(100, 2)

    for i in range (0,20):

        W1, W2 = train(path_Folder, WV, dim, W1, W2, eta, OutputFile, attOn, attScalar, activationFunc)

        if i%1 == 0:
            # print path_Folder_test
            # print path_Folder
            print "======================================================================"
            test(path_Folder, "train", WV, dim, W1, W2, OutputFile, attOn, attScalar, i, activationFunc)
            print "======================================================================"
            test(path_Folder_test, "test", WV, dim, W1, W2, OutputFile, attOn, attScalar, i, activationFunc)
            print "======================================================================"
'''

#Attention Weight



W1 = initialize_weight_variable(200, 100)
W2 = initialize_weight_variable(100, 2)
WSat = initialize_weight_variable(100, 100)
WNu = initialize_weight_variable(100, 100)

errors = []
#for i in range(0, 100):

    W1, W2, WSat, WNu = train_AttWeight(path_Folder, WV, dim, W1, W2, eta, OutputFile, WSat, WNu, activationFunc)
    #print W1 [0,:]
    if i % 1 == 0:
        error = calculateError_validation(path_Folder_val, "validation", WV, dim, W1, W2, OutputFile_val_err, WSat, WNu, i, activationFunc)
        errors.append(error)
        print "======================================================================"
        test_AttWeight_DrHarati(path_Folder, "train", WV, dim, W1, W2, OutputFile, WSat, WNu, i, activationFunc)
        print "======================================================================"
        test_AttWeight_DrHarati(path_Folder_val, "validation", WV, dim, W1, W2, OutputFile, WSat, WNu, i, activationFunc)
        print "======================================================================"
        test_AttWeight_DrHarati(path_Folder_test, "test", WV, dim, W1, W2, OutputFile, WSat, WNu, i, activationFunc)
        print "======================================================================"



OutputFile.close()
