"""

    Practica 4: LEARNING methods
    
    - Funcions per guardar, llegir i aprendre xarxes neurals i SVMs
    
    Aprenentatge Computacional - Universitat Autonoma de Barcelona
    
    Author: Jordi Gonzalez

"""

from numpy import *     # numpy, for maths computations
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
from pybrain.tools.shortcuts import buildNetwork
from sklearn.preprocessing import Imputer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer,TanhLayer, GaussianLayer, SoftmaxLayer, GateLayer
from pybrain.structure import FullConnection
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules import SoftmaxLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
import time


# guardar un model
def save_model(model_file_name, model):
    joblib.dump(model,model_file_name)


# carregar un model
def load_model(model_file_name):
        model = joblib.load(model_file_name)
        return model

def trainm_NN(x1,x2,parametresNN,actualNet):
        
        start_time = time.time()

        # prepare data  
        x1 = map(list,x1)
        x2 = map(list,x2) 
        X = x1+x2
        print shape(X)
        y1 = ones((shape(x1)[0],1))    # els positius son la classe '1'
        y2 = -1*ones((shape(x2)[0],1))    # els negatius son la classe '-1'
        Y = list(y1)+list(y2)
        Y = ravel(Y)

        #-----RED NUMERO 1          

        n1 = FeedForwardNetwork()
        
        inLayer = TanhLayer(199)
        hiddenLayer1=TanhLayer(40)
        hiddenLayer2 = LinearLayer(3)
        hiddenLayer3 = TanhLayer(5)
        hiddenLayer4 = LinearLayer(3)
        outLayer = LinearLayer(1)

        n1.addInputModule(inLayer)
        n1.addModule(hiddenLayer1)
        n1.addModule(hiddenLayer2)
        n1.addModule(hiddenLayer3)
        n1.addModule(hiddenLayer4)
        n1.addOutputModule(outLayer)

        in_to_hidden1 = FullConnection(inLayer, hiddenLayer1)
        hidden1_to_hidden2=FullConnection(hiddenLayer1,hiddenLayer2)
        hidden2_to_hidden3=FullConnection(hiddenLayer2,hiddenLayer3)
        hidden3_to_hidden4=FullConnection(hiddenLayer3,hiddenLayer4)
        hidden4_to_out = FullConnection(hiddenLayer4, outLayer)

        n1.addConnection(in_to_hidden1)
        n1.addConnection(hidden1_to_hidden2)
        n1.addConnection(hidden2_to_hidden3)
        n1.addConnection(hidden3_to_hidden4)
        n1.addConnection(hidden4_to_out)
            
        n1.sortModules() #Crida necesaria init de moduls interns

        #-----RED NUMERO 2          

        n2 = FeedForwardNetwork()
        
        inLayer = TanhLayer(199)
        hiddenLayer1=TanhLayer(12)
        hiddenLayer2 = LinearLayer(3)
        outLayer = LinearLayer(1)

        n2.addInputModule(inLayer)
        n2.addModule(hiddenLayer1)
        n2.addModule(hiddenLayer2)
        n2.addOutputModule(outLayer)

        in_to_hidden1 = FullConnection(inLayer, hiddenLayer1)
        hidden1_to_hidden2=FullConnection(hiddenLayer1,hiddenLayer2)
        hidden2_to_out = FullConnection(hiddenLayer2, outLayer)

        n2.addConnection(in_to_hidden1)
        n2.addConnection(hidden1_to_hidden2)
        n2.addConnection(hidden2_to_out)
            
        n2.sortModules() #Crida necesaria init de moduls interns

         #-----RED NUMERO 3          

        n3 = FeedForwardNetwork()
        
        inLayer = TanhLayer(199)
        hiddenLayer1=TanhLayer(27)
        hiddenLayer2 = TanhLayer(5)
        outLayer = LinearLayer(1)

        n3.addInputModule(inLayer)
        n3.addModule(hiddenLayer1)
        n3.addModule(hiddenLayer2)
        n3.addOutputModule(outLayer)

        in_to_hidden1 = FullConnection(inLayer, hiddenLayer1)
        hidden1_to_hidden2=FullConnection(hiddenLayer1,hiddenLayer2)
        hidden2_to_out = FullConnection(hiddenLayer2, outLayer)

        n3.addConnection(in_to_hidden1)
        n3.addConnection(hidden1_to_hidden2)
        n3.addConnection(hidden2_to_out)
            
        n3.sortModules() #Crida necesaria init de moduls interns
        
        #-----RED NUMERO 4 (implementar RELU)     

        #Selection of the net that gonna be trained
        if (actualNet==1):
            n=n1
        elif (actualNet==2):
            n=n2
        elif (actualNet==3):
            n=n3
        else:
            #TODO
            n=n1
        #Initialization weights
        if(actualNet==1):
            r = math.sqrt(6/((199+40+3+5+1)*(1.0)))
            sizeOfNet=199*40+40*3+3*5+5*3+3
        elif (actualNet==2):
            r = math.sqrt(6/((199+12+3+1)*(1.0)))
            sizeOfNet=199*12+12*3+3
        elif (actualNet==3):
            r = math.sqrt(6/((199+27+5+1)*(1.0)))
            sizeOfNet=199*27+27*5+5
        else:
            #TODO
            r = math.sqrt(6/((199+40+3+5+1)*(1.0)))
            sizeOfNet=199*40+40*3+3*5+5

        weights_init = random.uniform(low=-r, high=r, size=(sizeOfNet,))
        n._setParameters(weights_init)
        DS = ClassificationDataSet(199, nb_classes=1)
        for i in range(shape(X)[0]):
              DS.addSample(list(X[i]),Y[i])
 
        
        #DS._convertToOneOfMany() # No -> volem nomes una sortida
        #DS.setField('class', DS.getField('target'))
 
        trainer = BackpropTrainer(n, dataset=DS, momentum=parametresNN['Momentum'], learningrate=parametresNN['learningrate'], verbose=parametresNN['verbose'], weightdecay=parametresNN['weightdecay'], batchlearning=parametresNN['batchlearning'])
        trainningErrors,validationErrors=trainer.trainUntilConvergence(maxEpochs=parametresNN['maxEpochs'])
        f=trainer.trainEpochs(parametresNN['maxEpochs'])

        print "Red Activa: ",actualNet,"   Tiempo transcurrido: ",time.time() - start_time, "   Error final training:", trainningErrors[-1], "   Error final validation:", validationErrors[-1]
        return n
        
def relu(self, inbuf, outbuf):
        outbuf[:] = inbuf * (inbuf > 0)    
        
    # gravar un svm
def svm_save_model(model_file_name, model):
    joblib.dump(model,model_file_name)


# carregar un svm
def svm_load_model(model_file_name):
    model = joblib.load(model_file_name)
    return model
        
# train SVM
def train_SVM(x1,x2):
    # prepare data  
    x1 = map(list,x1)
    x2 = map(list,x2)
           
    X = x1+x2
    y1 = ones((shape(x1)[0],1))    # els positius son la classe '1'
    y2 = -1*ones((shape(x2)[0],1))    # els negatius son la classe '-1'
    Y = list(y1)+list(y2)
    Y = ravel(Y)

    svm = SVC(probability=True)      #Instantiating the SVM RBF classifier.
    params = {'C': [100,200]} #Defining the params C & Gamma which will be used by GridSearch. Param C does increase the weight of the 'fails'. Gamma does define the std of a gaussian.
    grid = GridSearchCV(svm, params, cv=5)    # internament fem un cross validation amb 5 conjunts
        
    # per evitar errors quan hi han mostres sense valor en algun dels 199 parametres que defineixen la mostra
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X)
    Imputer(axis=0, copy=True, missing_values='NaN', strategy='mean', verbose=0)   
    trainData = imp.transform(X)

    grid.fit(trainData, Y)        #Run fit with all sets of parameters.
    model = grid.best_estimator_
    


    return model

    


