'''
Text_Preprocessing
'''
import re
import os
import shutil

def preprocessor1(words):
    pre_word=re.sub(r"[^a-zA-Z]", " ", words.lower())
    return pre_word

def sepValidation(path_Folder):

    posFileList_main = os.listdir(path_Folder + "pos/")
    negFileList_main = os.listdir(path_Folder + "neg/")
    posFileList = []
    negFileList = []
    posFileList_val = []
    negFileList_val = []
    numberofSamples = min(len(posFileList_main), len(negFileList_main))
    for j in range(0, numberofSamples):
        if j%10==0:
            posFileList_val.append(posFileList_main[j])
            negFileList_val.append(negFileList_main[j])
        else :
            posFileList.append(posFileList_main[j])
            negFileList.append(negFileList_main[j])

    return posFileList, negFileList, posFileList_val, negFileList_val

def sepValidation_file(path_Folder_src, path_Folder_des):

    posFileList_main = os.listdir(path_Folder_src + "pos/")
    negFileList_main = os.listdir(path_Folder_src + "neg/")
    numberofSamples = min(len(posFileList_main), len(negFileList_main))

    for j in range(0, numberofSamples):
        if j % 10 == 0:
            shutil.move(path_Folder_src+ "pos/"+posFileList_main[j], path_Folder_des+"pos/")
            shutil.move(path_Folder_src+ "neg/"+negFileList_main[j], path_Folder_des+"neg/")

#sepValidation_file("/home/erfaneh/Shared/IMDB-Parsed/Train/", "/home/erfaneh/Shared/IMDB-Parsed/Evaluation/")