"""

    Practica 4: TESTING script
    
    - per generar el fitxer de prediccio amb que participar en la competicio
    
    Aprenentatge Computacional - Universitat Autonoma de Barcelona
    
    Author: Jordi Gonzalez

"""
from Practica4_NN import *
from sys import path
import multiprocessing as mp
from scipy.io import *
import numpy as np
from numpy import *
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score
import time
import os
path.append('.')
import pickle                    # save/load data files
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError

parametrosNN={
    'Dimensions':199,
    'Momentum': 0.9,
    'learningrate': 0.105,
    'verbose': True,
    'weightdecay': 0.01,
    'batchlearning': True,
    'maxEpochs': 35,
    'LrDecay': 1.0,
    'ValidationProportion' : 0.01
    #sobre els 100 segons (MacBook Pro, 2.6 GHz, RAM 8 GB)
    }




def test_image(hist, y_test):
    #hist: valors de la imatge actual
    #y_test: la classe real

    # Hi ha un classificador apres per cada classe
    code = zeros(int(nc))
  
    # one-vs-all: mirem la probabilitat de pertanyer a cada una de les 5 classes, i ens quedem amb el maxim    
    for cidx in range(nc):
        
        #fileNetwork = open(classifierDir+'NN_Dim'+str(parametrosNN['Dimensions'])+'_Epochs'+str(parametrosNN['maxEpochs'])+nom_classes[cidx]+'_vs_all.model', 'w')
        #NN_model = pickle.load(fileNetwork)
        
        NN_model = load_model(classifierDir+'NN_Dim'+str(parametrosNN['Dimensions'])+'_Epochs'+str(parametrosNN['maxEpochs'])+'_'+nom_classes[cidx]+'_vs_all.model')
        
        code[cidx]= NN_model.activate(hist)
       
    y_auto = int(argmax(code)+1)

    return (y_test, y_auto) 
    
    

def test_image_svm(hist, y_test):
    
    # Hi ha un SVM apres per cada classe
    code = zeros(int(nc))
             
    # one-vs-all: mirem la probabilitat de pertanyer a cada una de les 5 classes, i ens quedem amb el maxim    
    for cidx in range(nc):
        
        # carreguem el SVM per una classe: haureu de substituir aquest SVM per una xarxa neuronal
        svm_model = svm_load_model(classifierDir+'SVM_'+str(K)+'_'+nom_classes[cidx]+'_vs_all.model')         
            
        # mirem la probabilitat de que la mostra de test pertany i aquesta classe en concret, i la de que no pertanyi 
        decis =  svm_model.predict_proba(hist)   
    
        # hem de mirar que un 1 significa que pertany a la classe, 
        if svm_model.classes_[0] == 1:
            code[cidx] = decis[0][0] 
        else:
            code[cidx] = decis[0][1] 
                     
    # ens quedem amb la classe per la que el seu corresponent SVM doni una probabilitat maxima                                       
    y_auto = int(argmax(code)+1) 
            
    # retornem la classe predita i la classe real, per calcular la matriu de confusio i l'accuracy
    return (y_test, y_auto) 
    


# ******************  MAIN **************************
if __name__ == "__main__":
        
    # a canviar, depenent del vostre ordinador             
    DataOutputFolder =''

    classifierDir = DataOutputFolder + 'classifiers/'

    # aquest directori s'utilitzara per tal de treure el resultat de la competicio    
    testDataDir   = DataOutputFolder + 'TestData/'

    if not os.path.isdir(testDataDir):  os.mkdir(testDataDir)

    nc = 5 # number of classes
    nom_classes = ['car','dog','bicycle','motorbike','person']
    
    K = 199  # number of features per inmage
    nTestFiles = 50    # number of testing images per classe (al directori valData/ i Dataset/validation/)

    #Conjunt de Test, sense etiquetar 
    fx = open(testDataDir +'random_data_class'+'.pkl', 'rb')
    x_test_r = pickle.load(fx);
    fx.close()  
    
    y_predit = np.zeros(nc*nTestFiles)

    for c_test in range(len(x_test_r)):   # calculem els errors per totes les classes
           
        # hist conte la representacio de la imatge, es un vector de 199 dimensions           
        hist = x_test_r[c_test];
           
        (class_real_unknown, class_assigned) = test_image(hist, x_test_r[c_test])

        #Guardem la prediccio, que s'utilitzara per a la competicio
        y_predit[c_test]=class_assigned


    #Per obtenir la precisio en el conjunt de test, cal enviar aquest fitxer
    fp = open(testDataDir +'predictions_class'+'.pkl', 'wb')
    pickle.dump(y_predit,fp);
    fp.close()  

