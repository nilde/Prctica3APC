"""

    Practica 4: TRAINING script
    
    - codi per realitzar l'aprenentatge dels classificadors
    
    Aprenentatge Computacional - Universitat Autonoma de Barcelona
    
    Author: Jordi Gonzalez

"""
from Practica4_NN import *

from sys import path
from scipy.io import *
from numpy import *
import time
import os
path.append('.')        
import pickle                    # save/load data files
from pybrain.tools.shortcuts import buildNetwork


parametresNN={
    'Dimensions':199,
    'Momentum': 0.4,
    'learningrate': 0.078,
    'verbose': False,
    'weightdecay': 0.01,
    'batchlearning': True,
    'maxEpochs': 35,
    }


def train_NN(parametrosNN):
    # carreguem el fitxer que conte les dades d'aprenenatatge (200 mostres de 199 dimensions per cada classe)
    f = open(trainDataDir+'sift_sift_svm_traindata_K_'+str(K)+'.pkl', 'rb')
    trainData = pickle.load(f);
    trainLabels = pickle.load(f)
    f.close()
    
    for cidx in range(nc):
        # ens quedem amb les mostres de la classe que volem aprendre
        idx1 = where(trainLabels==cidx+1)
        x1 = trainData[idx1]

        # ens quedem amb totes les altres mostres de les altres classes, 
        # s'utilitzaran com a negatius per discriminar millor la classe que es vol aprendre 
        idx2 = where(trainLabels!=cidx+1)
        x2 = trainData[idx2]


        #APRENENTATGE XARXA NEURAL (a Practica4_NN.py)
        print 'TRAIN NN: Class '+str(nom_classes[cidx])+' vs All: '+str(shape(x1)[0])+' vs '+str(shape(x2)[0])+' samples \r' 
        momentumValues=arange(0.1,1.0,0.1)
        learningRateValues=arange(0.1,0.5,0.010)
        weightDecayValues=arange(0.1,0.5,0.05)
        epochsValues=arange(10,100,5)
        for momentum in momentumValues:
            for learningRate in learningRateValues:
                for weightDecay in weightDecayValues:
                    for epochs in epochsValues:
                        parametrosNN['Momentum']=momentum
                        parametrosNN['learningrate']=learningRate
                        parametrosNN['weightdecay']=weightDecay
                        parametrosNN['maxEpochs']=epochs
                        print 'Train con los valores',parametrosNN
                        try:
                            nn_model = trainm_NN(x1,x2,parametrosNN,1)
                        except Exception:
                            print "Error mas grande que 1(peta)"
                            pass
                        print '----------------------------------'
                        print '----------------------------------'


        # guardem la xarxa neural resultant
        fileNetwork = open(classifierDir+'NN_Dim'+str(parametrosNN['Dimensions'])+'_Epochs'+str(parametrosNN['maxEpochs'])+'_'+nom_classes[cidx]+'_vs_all.model', 'wb')
        pickle.dump(nn_model, fileNetwork)


        #PER COMPARAR: APRENENTATGE SVM (a Practica4_NN.py)
#        print 'TRAIN SVM: Class '+str(nom_classes[cidx])+' vs All: '+str(shape(x1)[0])+' vs '+str(shape(x2)[0])+' samples \r'                        
#        svm_model = train_SVM(x1,x2)   # crida a la funcio a practica4_svm.py

#        # guardem el model SVM apres per fer-lo servir al test: es troba a practica4_svm.py                 
#        svm_save_model(classifierDir+'SVM_'+str(K)+'_'+nom_classes[cidx]+'_vs_all.model', svm_model)




# ******************  MAIN **************************
if __name__ == "__main__":

    # a canviar, depenent del vostre ordinador                     
    # els directori on es troben:
    # 1) el classificador apres per cada classe el copiara a /classifiers/, 
    # 2) les dades d'aprenentatge, es a dir, un vector de 199 dimensions per cada imatge del training (trainData/),
    DataOutputFolder = ''
    classifierDir = DataOutputFolder + 'classifiers/'
    trainDataDir  = DataOutputFolder + 'trainData/'
    
    if not os.path.isdir(classifierDir): os.mkdir(classifierDir)
    if not os.path.isdir(trainDataDir):  os.mkdir(trainDataDir)

    nc = 5 # number of classes
    nom_classes = ['car','dog','bicycle','motorbike','person']
    
    K = 199 # number of features per inmage

    # ###############################
    #             TRAINING
    # ###############################    
    
    ts = time.time()
    train_NN(parametresNN)
    te = time.time()     
    print 'Learning finished in %2.2f sec' % (te-ts)
        
        
    
    



