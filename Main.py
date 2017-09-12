import numpy as np
from Method_WV import readWV
from Method_NeuralNets import initialize_weight_variable
from Method_RSTTree import readTree_att_Scalar
from Methods_Classification import train, test
import os
import math


from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

WV=readWV("/home/erfaneh/Shared/WordVector/glove.6B/glove.6B.100d.txt", stop)
path_Folder="/home/erfaneh/PycharmProjects/Recursive-neural-networks-TensorFlow/Data_Test/"
path_Folder_test = "/home/erfaneh/PycharmProjects/Recursive-neural-networks-TensorFlow/Data/"

posFileList = os.listdir(path_Folder+"pos/")
negFileList = os.listdir(path_Folder+"neg/")

numberofSamples = min (len(posFileList), len(negFileList))

eta = 0.005
dim = 100
OutputFile = open(path_Folder + "output.csv", "w")
OutputFile.write("attScalar, iteration, data, tp, tn, fp, fn, accuracy, precision, recall, F-1 measure\n")
attOn = "Nucleus"


for attScalar in np.arange(1.0, 2.0, 0.1):

    W1 = initialize_weight_variable(200, 100)
    W2 = initialize_weight_variable(100, 2)

    for i in range (0,20):

        W1, W2 = train(path_Folder, WV, dim, W1, W2, eta, OutputFile, attOn, attScalar)

        if i%1 == 0:
            test(path_Folder, "train", WV, dim, W1, W2, OutputFile, attOn, attScalar, i)
            test(path_Folder_test, "test", WV, dim, W1, W2, OutputFile, attOn, attScalar, i)

OutputFile.close()
