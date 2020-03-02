# 2020/2/26~. Earthquake epicenter placement

#%%##################################################################
#                                                                   #
#                    IMPORTING                                      #
#                                                                   #
#####################################################################

#\\\ Standard libraries:
import os
import numpy as np
import pickle as pkl
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import pickle
import datetime
from scipy.io import savemat

import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

#\\\ Own libraries:
import Utils.graphTools as graphTools
import Utils.dataTools
import Utils.graphML as gml
import Modules.architectures as archit
import Modules.model as model
import Modules.train_rnn_quake as train

#\\\ Separate functions:
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed

#%%##################################################################
#                                                                   #
#                    SETTING PARAMETERS                             #
#                                                                   #
#####################################################################
#
seqLen = 10 # length in seconds of the waves sampled at 100 Hz
downsampling = 50 # downsampling factor; new frequency is given by 100/downsampling
K = int(seqLen*100/downsampling) # number of samples in a sequence

thisFilename = 'Epicenter Estimation' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) # Dir where to save all
    # the results from each run

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + today
# Create directory
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
# Create the file where all the (hyper)parameters are results will be saved.
varsFile = os.path.join(saveDir,'hyperparameters.txt')
with open(varsFile, 'w+') as file:
    file.write('%s\n\n' % datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"))

#\\\ Save seeds for reproducibility
#   PyTorch seeds
torchState = torch.get_rng_state()
torchSeed = torch.initial_seed()
#   Numpy seeds
numpyState = np.random.RandomState().get_state()
#   Collect all random states
randomStates = []
randomStates.append({})
randomStates[0]['module'] = 'numpy'
randomStates[0]['state'] = numpyState
randomStates.append({})
randomStates[1]['module'] = 'torch'
randomStates[1]['state'] = torchState
randomStates[1]['seed'] = torchSeed
#   This list and dictionary follows the format to then be loaded, if needed,
#   by calling the loadSeed function in Utils.miscTools
saveSeed(randomStates, saveDir)

########
# DATA #
########
#
nNodes = 59 # Number of seismographs
nRegions = 11 # Number of regions in NZ

nTrain = 1648 # Number of training samples
nValid = 412 # Number of validation samples
nTest = 229 # Number of testing samples

#\\\ Save values:
writeVarValues(varsFile,
               {'nTrain': nTrain,
                'nValid': nValid,
                'nTest': nTest,
                'seqLen': seqLen,
                'downsampling': downsampling,
                'K': K})

############
# TRAINING #
############

#\\\ Individual model training options
trainer = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
learningRate = 0.00001 # In all options
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.CrossEntropyLoss() # This applies a softmax before feeding
    # it into the NLL, so we don't have to apply the softmax ourselves.

#\\\ Overall training options
nEpochs = 60 # Number of epochs
batchSize = 100 # Batch size
doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr
validationInterval = 10 # How many training steps to do the validation

#\\\ Save values
writeVarValues(varsFile,
               {'trainer': trainer,
                'learningRate': learningRate,
                'beta1': beta1,
                'lossFunction': lossFunction,
                'nEpochs': nEpochs,
                'batchSize': batchSize,
                'doLearningRateDecay': doLearningRateDecay,
                'learningRateDecayRate': learningRateDecayRate,
                'learningRateDecayPeriod': learningRateDecayPeriod,
                'validationInterval': validationInterval})

#################
# ARCHITECTURES #
#################

# Now that we have accumulated quite an extensive number of architectures, we
# might not want to run all of them. So here we select which ones we want to
# run.

# Select desired architectures
doSelectionGNN = True
doRNN_MLP = True
doGCRNN_MLP = True
doGCRNN_GNN =  True
doTimeGCRNN_MLP = True
doNodeGCRNN_MLP = True
doEdgeGCRNN_MLP = True

# In this section, we determine the (hyper)parameters of models that we are
# going to train. This only sets the parameters. The architectures need to be
# created later below. That is, any new architecture in this part, needs also
# to be coded later on. This is just to be easy to change the parameters once
# the architecture is created. Do not forget to add the name of the architecture
# to modelList.

modelList = []

# Parameters to share across several architectures

# Linear layer
F1 = 20 # Number of state features for the GCRNN architectures
K1 = 4 # Number of filter taps for all filters
rnnStateFeat = 21 # Number of state features for the RNN architecture

#\\\\\\\\\\\\
#\\\ MODEL 1: Selection GNN
#\\\\\\\\\\\\

if doSelectionGNN:

    hParamsSelGNNDeg = {}

    hParamsSelGNNDeg['name'] = 'Sel' # Name of the architecture

    #\\\ Architecture parameters
    hParamsSelGNNDeg['F'] = [K, 21] # Features per layer. For the GNN, the 
    # number of input features has to be the sequence length
    hParamsSelGNNDeg['K'] = [K1] # Number of filter taps per layer
    hParamsSelGNNDeg['bias'] = True # Decide whether to include a bias term
    hParamsSelGNNDeg['sigma'] = nn.ReLU # Selected nonlinearity
    hParamsSelGNNDeg['N'] = [nNodes] # Number of nodes to keep at the end of
    # each layer
    hParamsSelGNNDeg['rho'] = gml.NoPool # Summarizing function
    hParamsSelGNNDeg['alpha'] = [1] # alpha-hop neighborhood that
        # is affected by the summary
    hParamsSelGNNDeg['dimLayersMLP'] = [nRegions] # Dimension of the fully
        # connected layers after the GCN layers

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsSelGNNDeg)
    modelList += [hParamsSelGNNDeg['name']]
    
#\\\\\\\\\\\\
#\\\ MODEL 2: GCRNN followed by localized perceptrons
#\\\\\\\\\\\\

if doGCRNN_MLP:

    hParamsGCRNN_MLP = {} 

    hParamsGCRNN_MLP['name'] = 'GCRNNMLP' # Name of the architecture

    #\\\ Architecture parameters
    hParamsGCRNN_MLP['inFeatures'] = 1 
    hParamsGCRNN_MLP['stateFeatures'] = F1 
    hParamsGCRNN_MLP['inputFilterTaps'] = K1 
    hParamsGCRNN_MLP['stateFilterTaps'] = K1 
    hParamsGCRNN_MLP['stateNonlinearity'] = nn.functional.tanh
    hParamsGCRNN_MLP['outputNonlinearity'] = nn.ReLU
    hParamsGCRNN_MLP['dimLayersMLP'] = [nRegions] 
    hParamsGCRNN_MLP['bias'] = True
    hParamsGCRNN_MLP['time_gating'] = False
    hParamsGCRNN_MLP['spatial_gating'] = None

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsGCRNN_MLP)
    modelList += [hParamsGCRNN_MLP['name']]


#\\\\\\\\\\\\
#\\\ MODEL 3: RNN followed by MLP
#\\\\\\\\\\\\

if doRNN_MLP:

    hParamsRNN_MLP = {}

    hParamsRNN_MLP['name'] = 'RNNMLP' # Name of the architecture

    #\\\ Architecture parameters
    hParamsRNN_MLP['inFeatures'] = 1 # per node
    hParamsRNN_MLP['stateFeatures'] = rnnStateFeat
    hParamsRNN_MLP['stateNonlinearity'] = 'tanh'
    hParamsRNN_MLP['dimLayersMLP'] = [nRegions] 
    hParamsRNN_MLP['outputNonlinearity'] = nn.ReLU
    hParamsRNN_MLP['bias'] = True

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsRNN_MLP)
    modelList += [hParamsRNN_MLP['name']]

#\\\\\\\\\\\\
#\\\ MODEL 4: GCRNN followed by GNN
#\\\\\\\\\\\\

if doGCRNN_GNN:

    hParamsGCRNN_GNN = {}

    hParamsGCRNN_GNN['name'] = 'GCRNNGNN' # Name of the architecture

    #\\\ Architecture parameters
    hParamsGCRNN_GNN['inFeatures'] = 1 
    hParamsGCRNN_GNN['stateFeatures'] = F1
    hParamsGCRNN_GNN['inputFilterTaps'] = K1 
    hParamsGCRNN_GNN['stateFilterTaps'] = K1 
    hParamsGCRNN_GNN['stateNonlinearity'] = torch.tanh
    hParamsGCRNN_GNN['outputNonlinearity'] = nn.ReLU
    hParamsGCRNN_GNN['dimLayersMLP'] = [nRegions] 
    hParamsGCRNN_GNN['bias'] = True
    hParamsGCRNN_GNN['time_gating'] = False
    hParamsGCRNN_GNN['spatial_gating'] = None
    hParamsGCRNN_GNN['mlpType'] = 'oneMlp'
    hParamsGCRNN_GNN['finalNonlinearity'] = nn.ReLU
    hParamsGCRNN_GNN['dimNodeSignals'] = [F1,1]
    hParamsGCRNN_GNN['nFilterTaps'] = [K1]
    hParamsGCRNN_GNN['nSelectedNodes'] = [nNodes]
    hParamsGCRNN_GNN['poolingFunction'] = gml.NoPool
    hParamsGCRNN_GNN['poolingSize'] = [1]

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsGCRNN_GNN)
    modelList += [hParamsGCRNN_GNN['name']]

#\\\\\\\\\\\\
#\\\ MODEL 5: Time gated GCRNN followed by MLP
#\\\\\\\\\\\\

if doTimeGCRNN_MLP:

    hParamsTimeGCRNN_MLP = {}

    hParamsTimeGCRNN_MLP['name'] = 'TimeGCRNNMLP' # Name of the architecture

    #\\\ Architecture parameters
    hParamsTimeGCRNN_MLP['inFeatures'] = 1 
    hParamsTimeGCRNN_MLP['stateFeatures'] = F1 
    hParamsTimeGCRNN_MLP['inputFilterTaps'] = K1 
    hParamsTimeGCRNN_MLP['stateFilterTaps'] = K1 
    hParamsTimeGCRNN_MLP['stateNonlinearity'] = nn.functional.tanh
    hParamsTimeGCRNN_MLP['outputNonlinearity'] = nn.ReLU
    hParamsTimeGCRNN_MLP['dimLayersMLP'] = [nRegions] 
    hParamsTimeGCRNN_MLP['bias'] = True
    hParamsTimeGCRNN_MLP['time_gating'] = True
    hParamsTimeGCRNN_MLP['spatial_gating'] = None

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsTimeGCRNN_MLP)
    modelList += [hParamsTimeGCRNN_MLP['name']]

#\\\\\\\\\\\\
#\\\ MODEL 6: Node gated GCRNN followed by MLP
#\\\\\\\\\\\\

if doNodeGCRNN_MLP:

    hParamsNodeGCRNN_MLP = {} 

    hParamsNodeGCRNN_MLP['name'] = 'NodeGCRNNMLP' # Name of the architecture

    #\\\ Architecture parameters
    hParamsNodeGCRNN_MLP['inFeatures'] = 1 
    hParamsNodeGCRNN_MLP['stateFeatures'] = F1 
    hParamsNodeGCRNN_MLP['inputFilterTaps'] = K1 
    hParamsNodeGCRNN_MLP['stateFilterTaps'] = K1 
    hParamsNodeGCRNN_MLP['stateNonlinearity'] = nn.functional.tanh
    hParamsNodeGCRNN_MLP['outputNonlinearity'] = nn.ReLU
    hParamsNodeGCRNN_MLP['dimLayersMLP'] = [nRegions] 
    hParamsNodeGCRNN_MLP['bias'] = True
    hParamsNodeGCRNN_MLP['time_gating'] = False
    hParamsNodeGCRNN_MLP['spatial_gating'] = 'node'

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsNodeGCRNN_MLP)
    modelList += [hParamsNodeGCRNN_MLP['name']]

#\\\\\\\\\\\\
#\\\ MODEL 7: Edge gated GCRNN followed by MLP
#\\\\\\\\\\\\

if doEdgeGCRNN_MLP:

    hParamsEdgeGCRNN_MLP = {} 

    hParamsEdgeGCRNN_MLP['name'] = 'EdgeGCRNNMLP' # Name of the architecture

    #\\\ Architecture parameters
    hParamsEdgeGCRNN_MLP['inFeatures'] = 1 
    hParamsEdgeGCRNN_MLP['stateFeatures'] = F1 
    hParamsEdgeGCRNN_MLP['inputFilterTaps'] = K1 
    hParamsEdgeGCRNN_MLP['stateFilterTaps'] = K1 
    hParamsEdgeGCRNN_MLP['stateNonlinearity'] = nn.functional.tanh
    hParamsEdgeGCRNN_MLP['outputNonlinearity'] = nn.ReLU
    hParamsEdgeGCRNN_MLP['dimLayersMLP'] = [nRegions] 
    hParamsEdgeGCRNN_MLP['bias'] = True
    hParamsEdgeGCRNN_MLP['time_gating'] = False
    hParamsEdgeGCRNN_MLP['spatial_gating'] = 'edge'

    #\\\ Save Values:
    writeVarValues(varsFile, hParamsEdgeGCRNN_MLP)
    modelList += [hParamsEdgeGCRNN_MLP['name']]

###########
# LOGGING #
###########

# Options:
doPrint = True # Decide whether to print stuff while running
doLogging = False # Log into tensorboard
doSaveVars = True # Save (pickle) useful variables
doFigs = True # Plot some figures (this only works if doSaveVars is True)
# Parameters:
printInterval = 0 # After how many training steps, print the partial results
xAxisMultiplierTrain = 100 # How many training steps in between those shown in
    # the plot, i.e., one training step every xAxisMultiplierTrain is shown.
xAxisMultiplierValid = 10 # How many validation steps in between those shown,
    # same as above.

#\\\ Save values:
writeVarValues(varsFile,
               {'doPrint': doPrint,
                'doLogging': doLogging,
                'doSaveVars': doSaveVars,
                'doFigs': doFigs,
                'saveDir': saveDir,
                'printInterval': printInterval})

#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ Determine processing unit:
device = 'cpu' #'cuda:0'
# Notify:
if doPrint:
    print("Device selected: %s" % device)

#\\\ Logging options
if doLogging:
    from Utils.visualTools import Visualizer
    logsTB = os.path.join(saveDir, 'logsTB')
    logger = Visualizer(logsTB, name='visualResults')

#\\\ Save variables during evaluation.
# We will save all the evaluations obtained for each for the trained models.
# It basically is a dictionary, containing a list of lists. The key of the
# dictionary determines de the model, then the first list index determines
# which graph, and the second list index, determines which realization within
# that graph. Then, this will be converted to numpy to compute mean and standard
# deviation (across the graph dimension).
accBest = {} # Accuracy for the best model
accLast = {} # Accuracy for the last model
for thisModel in modelList: # Create an element for each graph realization,
    # each of these elements will later be another list for each realization.
    # That second list is created empty and just appends the results.
    accBest[thisModel] = [None] 
    accLast[thisModel] = [None] 

####################
# TRAINING OPTIONS #
####################

# Training phase. It has a lot of options that are input through a
# dictionary of arguments.
# The value of this options was decided above with the rest of the parameters.
# This just creates a dictionary necessary to pass to the train function.

trainingOptions = {}

if doLogging:
    trainingOptions['logger'] = logger
if doSaveVars:
    trainingOptions['saveDir'] = saveDir
if doPrint:
    trainingOptions['printInterval'] = printInterval
if doLearningRateDecay:
    trainingOptions['learningRateDecayRate'] = learningRateDecayRate
    trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
trainingOptions['validationInterval'] = validationInterval

nDataRealizations = 2 # number of times the training-valid.-test split is reshuffled


#%%##################################################################
#                                                                   #
#                    GRAPH REALIZATION                              #
#                                                                   #
#####################################################################

# Start generating a new graph for each of the number of graph realizations that
# we previously specified.

# The accBest and accLast variables, for each model, have a list with a
# total number of elements equal to the number of graphs we will generate
# Now, for each graph, we have multiple data realization, so we want, for
# each graph, to create a list to hold each of those values

for thisModel in modelList:
    accBest[thisModel] = []
    accLast[thisModel] = []

#%%##################################################################
#                                                                   #
#                    DATA HANDLING                                  #
#                                                                   #
#####################################################################

#########
# GRAPH #
#########

# Load seismograph network
Adj = pkl.load( open( "Adj.p", "rb" ) )
graphType = 'adjacency'
graphOptions = {}
graphOptions['adjacencyMatrix'] = Adj 
G = graphTools.Graph(graphType, nNodes, graphOptions)
G.computeGFT()

############
# DATASETS #
############

for realization in range(nDataRealizations):
#   Load earthquake dataset
    data = Utils.dataTools.QuakeData(nTrain, nValid, nTest, 
                                           seqLen, downsampling)
    data.astype(torch.float64)
    data.to(device)
    
    #%%##################################################################
    #                                                                   #
    #                    MODELS INITIALIZATION                          #
    #                                                                   #
    #####################################################################
    
    # This is the dictionary where we store the models (in a model.Model
    # class, that is then passed to training).
    modelsGNN = {}
    
    # If a new model is to be created, it should be called for here.
    
    #%%\\\\\\\\\\
    #\\\ MODEL 1: Selection GNN ordered by Degree
    #\\\\\\\\\\\\
    
    if doSelectionGNN:
    
        thisName = hParamsSelGNNDeg['name']
    
        # If more than one graph or data realization is going to be carried
        # out, we are going to store all of thos models separately, so that
        # any of them can be brought back and studied in detail.
        if nDataRealizations > 1:
            thisName += 'R%02d' % realization
    
        ##############
        # PARAMETERS #
        ##############
    
        #\\\ Optimizer options
        #   (If different from the default ones, change here.)
        thisTrainer = trainer
        thisLearningRate = learningRate
        thisBeta1 = beta1
        thisBeta2 = beta2
        #\\\ Ordering/GSO
        S = G.S/np.abs(np.max(np.diag(G.E)))
        S = np.expand_dims(S,axis=0)
        order = np.arange(G.N)
        # order is an np.array with the ordering of the nodes with respect
        # to the original GSO (the original GSO is kept in G.S).
    
        ################
        # ARCHITECTURE #
        ################
    
        thisArchit = archit.SelectionGNN(# Graph filtering
                                         hParamsSelGNNDeg['F'],
                                         hParamsSelGNNDeg['K'],
                                         hParamsSelGNNDeg['bias'],
                                         # Nonlinearity
                                         hParamsSelGNNDeg['sigma'],
                                         # Pooling
                                         hParamsSelGNNDeg['N'],
                                         hParamsSelGNNDeg['rho'],
                                         hParamsSelGNNDeg['alpha'],
                                         # MLP
                                         hParamsSelGNNDeg['dimLayersMLP'],
                                         # Structure
                                         S)
        # This is necessary to move all the learnable parameters to be
        # stored in the device (mostly, if it's a GPU)
        thisArchit.to(device)
    
        #############
        # OPTIMIZER #
        #############
    
        if thisTrainer == 'ADAM':
            thisOptim = optim.Adam(thisArchit.parameters(),
                                   lr = learningRate, betas = (beta1,beta2))
        elif thisTrainer == 'SGD':
            thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
        elif thisTrainer == 'RMSprop':
            thisOptim = optim.RMSprop(thisArchit.parameters(),
                                      lr = learningRate, alpha = beta1)
    
        ########
        # LOSS #
        ########
    
        thisLossFunction = lossFunction # (if different from default, change
                                        # it here)
    
        #########
        # MODEL #
        #########
    
        SelGNNDeg = model.Model(thisArchit, thisLossFunction, thisOptim,
                                thisName, saveDir, order)
    
        modelsGNN[thisName] = SelGNNDeg
    
        writeVarValues(varsFile,
                       {'name': thisName,
                        'thisTrainer': thisTrainer,
                        'thisLearningRate': thisLearningRate,
                        'thisBeta1': thisBeta1,
                        'thisBeta2': thisBeta2})
    
    #%%\\\\\\\\\\
    #\\\ MODEL 2: GCRNN + MLP
    #\\\\\\\\\\\\
    
    if doGCRNN_MLP:
    
        thisName = hParamsGCRNN_MLP['name']
    
        # If more than one graph or data realization is going to be carried
        # out, we are going to store all of thos models separately, so that
        # any of them can be brought back and studied in detail.
        if nDataRealizations > 1:
            thisName += 'R%02d' % realization
    
        ##############
        # PARAMETERS #
        ##############
    
        #\\\ Optimizer options
        #   (If different from the default ones, change here.)
        thisTrainer = trainer
        thisLearningRate = learningRate
        thisBeta1 = beta1
        thisBeta2 = beta2
    
        #\\\ Ordering/GSO
        S = G.S/np.abs(np.max(np.diag(G.E)))
        S = np.expand_dims(S,axis=0)
        order = np.arange(G.N)
    
        ################
        # ARCHITECTURE #
        ################
    
        thisArchit = archit.GatedGCRNNforClassification(hParamsGCRNN_MLP['inFeatures'],
                                         hParamsGCRNN_MLP['stateFeatures'],
                                         hParamsGCRNN_MLP['inputFilterTaps'],
                                         hParamsGCRNN_MLP['stateFilterTaps'],
                                         hParamsGCRNN_MLP['stateNonlinearity'],
                                         hParamsGCRNN_MLP['outputNonlinearity'],
                                         hParamsGCRNN_MLP['dimLayersMLP'],
                                         S,
                                         hParamsGCRNN_MLP['bias'],
                                         hParamsGCRNN_MLP['time_gating'],
                                         hParamsGCRNN_MLP['spatial_gating'])
        # This is necessary to move all the learnable parameters to be
        # stored in the device (mostly, if it's a GPU)
        thisArchit.to(device)
    
        #############
        # OPTIMIZER #
        #############
    
        if thisTrainer == 'ADAM':
            thisOptim = optim.Adam(thisArchit.parameters(),
                                   lr = learningRate, betas = (beta1,beta2))
        elif thisTrainer == 'SGD':
            thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
        elif thisTrainer == 'RMSprop':
            thisOptim = optim.RMSprop(thisArchit.parameters(),
                                      lr = learningRate, alpha = beta1)
    
        ########
        # LOSS #
        ########
    
        thisLossFunction = lossFunction # (if different from default, change
                                        # it here)
    
        #########
        # MODEL #
        #########
    
        GCRNN_MLP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                thisName, saveDir, order)
    
        modelsGNN[thisName] = GCRNN_MLP
    
        writeVarValues(varsFile,
                       {'name': thisName,
                        'thisTrainer': thisTrainer,
                        'thisLearningRate': thisLearningRate,
                        'thisBeta1': thisBeta1,
                        'thisBeta2': thisBeta2})
    
     
    #%%\\\\\\\\\\
    #\\\ MODEL 3: RNN + MLP
    #\\\\\\\\\\\\
    
    if doRNN_MLP:
    
        thisName = hParamsRNN_MLP['name']
    
        # If more than one graph or data realization is going to be carried
        # out, we are going to store all of thos models separately, so that
        # any of them can be brought back and studied in detail.
        if nDataRealizations > 1:
            thisName += 'R%02d' % realization
    
        ##############
        # PARAMETERS #
        ##############
    
        #\\\ Optimizer options
        #   (If different from the default ones, change here.)
        thisTrainer = trainer
        thisLearningRate = learningRate
        thisBeta1 = beta1
        thisBeta2 = beta2
    
        #\\\ Ordering/GSO
        S = G.S/np.abs(np.max(np.diag(G.E)))
        S = np.expand_dims(S,axis=0)
        order = np.arange(G.N)
    
        ################
        # ARCHITECTURE #
        ################
    
        thisArchit = archit.RNNforClassification(hParamsRNN_MLP['inFeatures'],
                                         hParamsRNN_MLP['stateFeatures'],
                                         hParamsRNN_MLP['stateNonlinearity'],
                                         hParamsRNN_MLP['dimLayersMLP'],
                                         hParamsRNN_MLP['outputNonlinearity'],
                                         S,
                                         hParamsRNN_MLP['bias'])
        # This is necessary to move all the learnable parameters to be
        # stored in the device (mostly, if it's a GPU)
        thisArchit.to(device)
    
        #############
        # OPTIMIZER #
        #############
    
        if thisTrainer == 'ADAM':
            thisOptim = optim.Adam(thisArchit.parameters(),
                                   lr = learningRate, betas = (beta1,beta2))
        elif thisTrainer == 'SGD':
            thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
        elif thisTrainer == 'RMSprop':
            thisOptim = optim.RMSprop(thisArchit.parameters(),
                                      lr = learningRate, alpha = beta1)
    
        ########
        # LOSS #
        ########
    
        thisLossFunction = lossFunction # (if different from default, change
                                        # it here)
    
        #########
        # MODEL #
        #########
    
        GCRNN_MLP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                thisName, saveDir, order)
    
        modelsGNN[thisName] = GCRNN_MLP
    
        writeVarValues(varsFile,
                       {'name': thisName,
                        'thisTrainer': thisTrainer,
                        'thisLearningRate': thisLearningRate,
                        'thisBeta1': thisBeta1,
                        'thisBeta2': thisBeta2})      
    
    #%%\\\\\\\\\\
    #\\\ MODEL 4: GCRNN + GNN
    #\\\\\\\\\\\\
    
    if doGCRNN_GNN:
    
        thisName = hParamsGCRNN_GNN['name']
    
        # If more than one graph or data realization is going to be carried
        # out, we are going to store all of thos models separately, so that
        # any of them can be brought back and studied in detail.
        if nDataRealizations > 1:
            thisName += 'R%02d' % realization
    
        ##############
        # PARAMETERS #
        ##############
    
        #\\\ Optimizer options
        #   (If different from the default ones, change here.)
        thisTrainer = trainer
        thisLearningRate = learningRate
        thisBeta1 = beta1
        thisBeta2 = beta2
    
        #\\\ Ordering/GSO
        S = G.S/np.abs(np.max(np.diag(G.E)))
        S = np.expand_dims(S,axis=0)
        order = np.arange(G.N)
    
        ################
        # ARCHITECTURE #
        ################
    
        thisArchit = archit.GatedGCRNNforClassification(hParamsGCRNN_GNN['inFeatures'],
                                         hParamsGCRNN_GNN['stateFeatures'],
                                         hParamsGCRNN_GNN['inputFilterTaps'],
                                         hParamsGCRNN_GNN['stateFilterTaps'],
                                         hParamsGCRNN_GNN['stateNonlinearity'],
                                         hParamsGCRNN_GNN['outputNonlinearity'],
                                         hParamsGCRNN_GNN['dimLayersMLP'],
                                         S,
                                         hParamsGCRNN_GNN['bias'],
                                         hParamsGCRNN_GNN['time_gating'],
                                         hParamsGCRNN_GNN['spatial_gating'],
                                         hParamsGCRNN_GNN['finalNonlinearity'],	
                                         hParamsGCRNN_GNN['dimNodeSignals'],
                                         hParamsGCRNN_GNN['nFilterTaps'],
                                         hParamsGCRNN_GNN['nSelectedNodes'],
                                         hParamsGCRNN_GNN['poolingFunction'],
                                         hParamsGCRNN_GNN['poolingSize'])
        # This is necessary to move all the learnable parameters to be
        # stored in the device (mostly, if it's a GPU)
        thisArchit.to(device)
    
        #############
        # OPTIMIZER #
        #############
    
        if thisTrainer == 'ADAM':
            thisOptim = optim.Adam(thisArchit.parameters(),
                                   lr = learningRate, betas = (beta1,beta2))
        elif thisTrainer == 'SGD':
            thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
        elif thisTrainer == 'RMSprop':
            thisOptim = optim.RMSprop(thisArchit.parameters(),
                                      lr = learningRate, alpha = beta1)
    
        ########
        # LOSS #
        ########
    
        thisLossFunction = lossFunction # (if different from default, change
                                        # it here)
    
        #########
        # MODEL #
        #########
    
        GCRNN_GNN = model.Model(thisArchit, thisLossFunction, thisOptim,
                                thisName, saveDir, order)
    
        modelsGNN[thisName] = GCRNN_GNN
    
        writeVarValues(varsFile,
                       {'name': thisName,
                        'thisTrainer': thisTrainer,
                        'thisLearningRate': thisLearningRate,
                        'thisBeta1': thisBeta1,
                        'thisBeta2': thisBeta2})
    
    #%%\\\\\\\\\\
    #\\\ MODEL 5: Time Gated GCRNN + MLP
    #\\\\\\\\\\\\
    
    if doTimeGCRNN_MLP:
    
        thisName = hParamsTimeGCRNN_MLP['name']
    
        # If more than one graph or data realization is going to be carried
        # out, we are going to store all of thos models separately, so that
        # any of them can be brought back and studied in detail.
        if nDataRealizations > 1:
            thisName += 'R%02d' % realization
    
        ##############
        # PARAMETERS #
        ##############
    
        #\\\ Optimizer options
        #   (If different from the default ones, change here.)
        thisTrainer = trainer
        thisLearningRate = learningRate
        thisBeta1 = beta1
        thisBeta2 = beta2
    
        #\\\ Ordering/GSO
        S = G.S/np.abs(np.max(np.diag(G.E)))
        S = np.expand_dims(S,axis=0)
        order = np.arange(G.N)
    
        ################
        # ARCHITECTURE #
        ################
    
        thisArchit = archit.GatedGCRNNforClassification(hParamsTimeGCRNN_MLP['inFeatures'],
                                         hParamsTimeGCRNN_MLP['stateFeatures'],
                                         hParamsTimeGCRNN_MLP['inputFilterTaps'],
                                         hParamsTimeGCRNN_MLP['stateFilterTaps'],
                                         hParamsTimeGCRNN_MLP['stateNonlinearity'],
                                         hParamsTimeGCRNN_MLP['outputNonlinearity'],
                                         hParamsTimeGCRNN_MLP['dimLayersMLP'],
                                         S,
                                         hParamsTimeGCRNN_MLP['bias'],
                                         hParamsTimeGCRNN_MLP['time_gating'],
                                         hParamsTimeGCRNN_MLP['spatial_gating'])
        # This is necessary to move all the learnable parameters to be
        # stored in the device (mostly, if it's a GPU)
        thisArchit.to(device)
    
        #############
        # OPTIMIZER #
        #############
    
        if thisTrainer == 'ADAM':
            thisOptim = optim.Adam(thisArchit.parameters(),
                                   lr = learningRate, betas = (beta1,beta2))
        elif thisTrainer == 'SGD':
            thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
        elif thisTrainer == 'RMSprop':
            thisOptim = optim.RMSprop(thisArchit.parameters(),
                                      lr = learningRate, alpha = beta1)
    
        ########
        # LOSS #
        ########
    
        thisLossFunction = lossFunction # (if different from default, change
                                        # it here)
    
        #########
        # MODEL #
        #########
    
        TimeGCRNN_MLP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                thisName, saveDir, order)
    
        modelsGNN[thisName] = TimeGCRNN_MLP
    
        writeVarValues(varsFile,
                       {'name': thisName,
                        'thisTrainer': thisTrainer,
                        'thisLearningRate': thisLearningRate,
                        'thisBeta1': thisBeta1,
                        'thisBeta2': thisBeta2})
    
    #%%\\\\\\\\\\
    #\\\ MODEL 6: Node Gated GCRNN + MLP
    #\\\\\\\\\\\\
    
    if doNodeGCRNN_MLP:
    
        thisName = hParamsNodeGCRNN_MLP['name']
    
        # If more than one graph or data realization is going to be carried
        # out, we are going to store all of thos models separately, so that
        # any of them can be brought back and studied in detail.
        if nDataRealizations > 1:
            thisName += 'R%02d' % realization
    
        ##############
        # PARAMETERS #
        ##############
    
        #\\\ Optimizer options
        #   (If different from the default ones, change here.)
        thisTrainer = trainer
        thisLearningRate = learningRate
        thisBeta1 = beta1
        thisBeta2 = beta2
    
        #\\\ Ordering/GSO
        S = G.S/np.abs(np.max(np.diag(G.E)))
        S = np.expand_dims(S,axis=0)
        order = np.arange(G.N)
    
        ################
        # ARCHITECTURE #
        ################
    
        thisArchit = archit.GatedGCRNNforClassification(hParamsNodeGCRNN_MLP['inFeatures'],
                                         hParamsNodeGCRNN_MLP['stateFeatures'],
                                         hParamsNodeGCRNN_MLP['inputFilterTaps'],
                                         hParamsNodeGCRNN_MLP['stateFilterTaps'],
                                         hParamsNodeGCRNN_MLP['stateNonlinearity'],
                                         hParamsNodeGCRNN_MLP['outputNonlinearity'],
                                         hParamsNodeGCRNN_MLP['dimLayersMLP'],
                                         S,
                                         hParamsNodeGCRNN_MLP['bias'],
                                         hParamsNodeGCRNN_MLP['time_gating'],
                                         hParamsNodeGCRNN_MLP['spatial_gating'])
        # This is necessary to move all the learnable parameters to be
        # stored in the device (mostly, if it's a GPU)
        thisArchit.to(device)
    
        #############
        # OPTIMIZER #
        #############
    
        if thisTrainer == 'ADAM':
            thisOptim = optim.Adam(thisArchit.parameters(),
                                   lr = learningRate, betas = (beta1,beta2))
        elif thisTrainer == 'SGD':
            thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
        elif thisTrainer == 'RMSprop':
            thisOptim = optim.RMSprop(thisArchit.parameters(),
                                      lr = learningRate, alpha = beta1)
    
        ########
        # LOSS #
        ########
    
        thisLossFunction = lossFunction # (if different from default, change
                                        # it here)
    
        #########
        # MODEL #
        #########
    
        NodeGCRNN_MLP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                thisName, saveDir, order)
    
        modelsGNN[thisName] = NodeGCRNN_MLP
    
        writeVarValues(varsFile,
                       {'name': thisName,
                        'thisTrainer': thisTrainer,
                        'thisLearningRate': thisLearningRate,
                        'thisBeta1': thisBeta1,
                        'thisBeta2': thisBeta2})
    
     #%%\\\\\\\\\\
    #\\\ MODEL 7: Edge Gated GCRNN + MLP
    #\\\\\\\\\\\\
    
    if doEdgeGCRNN_MLP:
    
        thisName = hParamsEdgeGCRNN_MLP['name']
    
        # If more than one graph or data realization is going to be carried
        # out, we are going to store all of thos models separately, so that
        # any of them can be brought back and studied in detail.
        if nDataRealizations > 1:
            thisName += 'R%02d' % realization
    
        ##############
        # PARAMETERS #
        ##############
    
        #\\\ Optimizer options
        #   (If different from the default ones, change here.)
        thisTrainer = trainer
        thisLearningRate = learningRate
        thisBeta1 = beta1
        thisBeta2 = beta2
    
        #\\\ Ordering/GSO
        S = G.S/np.abs(np.max(np.diag(G.E)))
        S = np.expand_dims(S,axis=0)
        order = np.arange(G.N)
    
        ################
        # ARCHITECTURE #
        ################
    
        thisArchit = archit.GatedGCRNNforClassification(hParamsEdgeGCRNN_MLP['inFeatures'],
                                         hParamsEdgeGCRNN_MLP['stateFeatures'],
                                         hParamsEdgeGCRNN_MLP['inputFilterTaps'],
                                         hParamsEdgeGCRNN_MLP['stateFilterTaps'],
                                         hParamsEdgeGCRNN_MLP['stateNonlinearity'],
                                         hParamsEdgeGCRNN_MLP['outputNonlinearity'],
                                         hParamsEdgeGCRNN_MLP['dimLayersMLP'],
                                         S,
                                         hParamsEdgeGCRNN_MLP['bias'],
                                         hParamsEdgeGCRNN_MLP['time_gating'],
                                         hParamsEdgeGCRNN_MLP['spatial_gating'])
        # This is necessary to move all the learnable parameters to be
        # stored in the device (mostly, if it's a GPU)
        thisArchit.to(device)
    
        #############
        # OPTIMIZER #
        #############
    
        if thisTrainer == 'ADAM':
            thisOptim = optim.Adam(thisArchit.parameters(),
                                   lr = learningRate, betas = (beta1,beta2))
        elif thisTrainer == 'SGD':
            thisOptim = optim.SGD(thisArchit.parameters(), lr=learningRate)
        elif thisTrainer == 'RMSprop':
            thisOptim = optim.RMSprop(thisArchit.parameters(),
                                      lr = learningRate, alpha = beta1)
    
        ########
        # LOSS #
        ########
    
        thisLossFunction = lossFunction # (if different from default, change
                                        # it here)
    
        #########
        # MODEL #
        #########
    
        EdgeGCRNN_MLP = model.Model(thisArchit, thisLossFunction, thisOptim,
                                thisName, saveDir, order)
    
        modelsGNN[thisName] = EdgeGCRNN_MLP
    
        writeVarValues(varsFile,
                       {'name': thisName,
                        'thisTrainer': thisTrainer,
                        'thisLearningRate': thisLearningRate,
                        'thisBeta1': thisBeta1,
                        'thisBeta2': thisBeta2})
    
    #%%##################################################################
    #                                                                   #
    #                    TRAINING                                       #
    #                                                                   #
    #####################################################################
    
    
    ############
    # TRAINING #
    ############
    
    # On top of the rest of the training options, we pass the identification
    # of this specific graph/data realization.
    
    if nDataRealizations > 1:
        trainingOptions['realizationNo'] = realization
    
    #        h0 = torch.zeros(batchSize,seqLen,nNodes)
    #        c0 = h0
    
    # This is the function that trains the models detailed in the dictionary
    # modelsGNN using the data, with the specified training options.
    train.MultipleModels(modelsGNN, data,
                         nEpochs = nEpochs, batchSize = batchSize, seqLen = K, 
                         stateFeat = F1, rnnStateFeat = rnnStateFeat,
                         **trainingOptions)

    #%%##################################################################
    #                                                                   #
    #                    EVALUATION                                     #
    #                                                                   #
    #####################################################################
    
    # Now that the model has been trained, we evaluate them on the test
    # samples.
    
    # We have two versions of each model to evaluate: the one obtained
    # at the best result of the validation step, and the last trained model.
    
    ########
    # DATA #
    ########
    
    xTest, yTest = data.getSamples('test')
    xTest = xTest.view(nTest,K,-1)
    yTest = yTest.view(nTest,-1)
    
    ##############
    # BEST MODEL #
    ##############
    
    if doPrint:
        print("\n Total test accuracy (Best):", flush = True)
    
    for key in modelsGNN.keys():
        # Update order and adapt dimensions (this data has one input feature,
        # so we need to add that dimension; make it from B x N to B x F x N)
        xTestOrdered = xTest[:,:,modelsGNN[key].order]
        if 'RNN' in modelsGNN[key].name or 'rnn' in modelsGNN[key].name or \
                                                'Rnn' in modelsGNN[key].name:
            xTestOrdered = xTestOrdered.unsqueeze(2)
    
        else:
                xTestOrdered = xTestOrdered.view(nTest,K,-1)
    
        with torch.no_grad():
            # Process the samples
            if 'GCRNN' in modelsGNN[key].name or 'gcrnn' in modelsGNN[key].name or \
                                                'GCRnn' in modelsGNN[key].name:
                h0t = torch.zeros(nTest,F1,nNodes)
                yHatTest = modelsGNN[key].archit(xTestOrdered,h0t)
            elif 'RNN' in modelsGNN[key].name or 'rnn' in modelsGNN[key].name or \
                                                'Rnn' in modelsGNN[key].name:
                h0t = torch.zeros(nTest,rnnStateFeat)
                c0t = h0t
                yHatTest = modelsGNN[key].archit(xTestOrdered,h0t,c0t)
            else:
                yHatTest = modelsGNN[key].archit(xTestOrdered)
            # We compute the accuracy
            thisAccBest = data.evaluate(yHatTest, yTest.squeeze())
    
        if doPrint:
            print("%s: %4.2f %%" % (key, thisAccBest*100 ), flush = True)
    
        # Save value
        writeVarValues(varsFile,
                   {'accBest%s' % key: thisAccBest})
    
        # Now check which is the model being trained
        for thisModel in modelList:
            # If the name in the modelList is contained in the name with
            # the key, then that's the model, and save it
            # For example, if 'SelGNNDeg' is in thisModelList, then the
            # correct key will read something like 'SelGNNDegG01R00' so
            # that's the one to save.
            if thisModel in key:
                accBest[thisModel] += [thisAccBest.item()]
            # This is so that we can later compute a total accuracy with
            # the corresponding error.
    
    ##############
    # LAST MODEL #
    ##############
    
    # And repeat for the last model
    
    if doPrint:
        print("\n Total test accuracy (Last):", flush = True)
    
    # Update order and adapt dimensions
    for key in modelsGNN.keys():
        modelsGNN[key].load(label = 'Last')
        xTestOrdered = xTest[:,:,modelsGNN[key].order]
    
        xTestOrdered = xTest[:,:,modelsGNN[key].order]
        if 'RNN' in modelsGNN[key].name or 'rnn' in modelsGNN[key].name or \
                                            'Rnn' in modelsGNN[key].name:
            xTestOrdered = xTestOrdered.unsqueeze(2)        
        
        else:
            xTestOrdered = xTestOrdered.view(nTest,K,-1)
        
        with torch.no_grad():
            # Process the samples
            if 'GCRNN' in modelsGNN[key].name or 'gcrnn' in modelsGNN[key].name or \
                                                'GCRnn' in modelsGNN[key].name:
                h0t = torch.zeros(nTest,F1,nNodes)
                yHatTest = modelsGNN[key].archit(xTestOrdered,h0t)
            elif 'RNN' in modelsGNN[key].name or 'rnn' in modelsGNN[key].name or \
                                                'Rnn' in modelsGNN[key].name:
                h0t = torch.zeros(nTest,rnnStateFeat)
                c0t = h0t
                yHatTest = modelsGNN[key].archit(xTestOrdered,h0t,c0t)
            else:
                yHatTest = modelsGNN[key].archit(xTestOrdered)
    
            # We compute the accuracy
            thisAccLast = data.evaluate(yHatTest, yTest.squeeze())
    
        if doPrint:
            print("%s: %4.2f %%" % (key, thisAccLast*100), flush = True)
    
        # Save values:
        writeVarValues(varsFile,
                   {'accLast%s' % key: thisAccLast})
        # And repeat for the last model:
        for thisModel in modelList:
            if thisModel in key:
                accLast[thisModel] += [thisAccLast.item()]

############################
# FINAL EVALUATION RESULTS #
############################

# Now that we have computed the accuracy of all runs, we can obtain a final
# result (mean and standard deviation)

meanAccBest = {} # Mean across graphs (after having averaged across data
    # realizations)
meanAccLast = {} # Mean across graphs
stdDevAccBest = {} # Standard deviation across graphs
stdDevAccLast = {} # Standard deviation across graphs

if doPrint:
    print("\nFinal evaluations")

for thisModel in modelList:
    # Convert the lists into a nGraphRealizations x nDataRealizations matrix
    accBest[thisModel] = np.array(accBest[thisModel])
    accLast[thisModel] = np.array(accLast[thisModel])

    # And now compute the statistics (across graphs)
    meanAccBest[thisModel] = np.mean(accBest[thisModel])
    meanAccLast[thisModel] = np.mean(accLast[thisModel])
    stdDevAccBest[thisModel] = np.std(accBest[thisModel])
    stdDevAccLast[thisModel] = np.std(accLast[thisModel])

    # And print it:
    if doPrint:
        print("\t%s: %6.2f%% (+-%6.2f%%) [Best] %6.2f%% (+-%6.2f%%) [Last]" % (
                thisModel,
                100*meanAccBest[thisModel],
                100*stdDevAccBest[thisModel],
                100*meanAccLast[thisModel],
                100*stdDevAccLast[thisModel]))

    # Save values
    writeVarValues(varsFile,
               {'meanAccBest%s' % thisModel: meanAccBest[thisModel],
                'stdDevAccBest%s' % thisModel: stdDevAccBest[thisModel],
                'meanAccLast%s' % thisModel: meanAccLast[thisModel],
                'stdDevAccLast%s' % thisModel : stdDevAccLast[thisModel]})

#%%##################################################################
#                                                                   #
#                    PLOT                                           #
#                                                                   #
#####################################################################

# Finally, we might want to plot several quantities of interest

if doFigs and doSaveVars:

    ###################
    # DATA PROCESSING #
    ###################

    # Again, we have training and validation metrics (loss and accuracy
    # -evaluation-) for many runs, so we need to carefully load them and compute
    # the relevant statistics from these realizations.

    #\\\ SAVE SPACE:
    # Create the variables to save all the realizations. This is, again, a
    # dictionary, where each key represents a model, and each model is a list
    # of lists, one list for each graph, and one list for each data realization.
    # Each data realization, in this case, is not a scalar, but a vector of
    # length the number of training steps (or of validation steps)
    lossTrain = {}
    evalTrain = {}
    lossValid = {}
    evalValid = {}
    # Initialize the graph dimension
    for thisModel in modelList:
        lossTrain[thisModel] = [None] 
        evalTrain[thisModel] = [None] 
        lossValid[thisModel] = [None] 
        evalValid[thisModel] = [None] 
        # Initialize the data realization dimension with empty lists to then
        # append each realization when we load it.
        lossTrain[thisModel] = []
        evalTrain[thisModel] = []
        lossValid[thisModel] = []
        evalValid[thisModel] = []

    #\\\ FIGURES DIRECTORY:
    saveDirFigs = os.path.join(saveDir,'figs')
    # If it doesn't exist, create it.
    if not os.path.exists(saveDirFigs):
        os.makedirs(saveDirFigs)

    #\\\ LOAD DATA:
    # Path where the saved training variables should be
    pathToTrainVars = os.path.join(saveDir,'trainVars')
    # Get all the training files:
    allTrainFiles = next(os.walk(pathToTrainVars))[2]
    # Go over each of them (this can't be empty since we are also checking for
    # doSaveVars to be true, what guarantees that the variables have been saved.
    for file in allTrainFiles:
        # Check that it is a pickle file
        if '.pkl' in file:
            # Open the file
            with open(os.path.join(pathToTrainVars,file),'rb') as fileTrainVars:
                # Load it
                thisVarsDict = pickle.load(fileTrainVars)
                # store them
                nBatches = thisVarsDict['nBatches']
                thisLossTrain = thisVarsDict['lossTrain']
                thisEvalTrain = thisVarsDict['evalTrain']
                thisLossValid = thisVarsDict['lossValid']
                thisEvalValid = thisVarsDict['evalValid']
                if 'realizationNo' in thisVarsDict.keys():
                    thisR = thisVarsDict['realizationNo']
                else:
                    thisR = 0
                # And add them to the corresponding variables
                for key in thisLossTrain.keys():
                # This part matches each realization (saved with a different
                # name due to identification of graph and data realization) with
                # the specific model.
                    for thisModel in modelList:
                        if thisModel in key:
                            lossTrain[thisModel] += [thisLossTrain[key]]
                            evalTrain[thisModel] += [thisEvalTrain[key]]
                            lossValid[thisModel] += [thisLossValid[key]]
                            evalValid[thisModel] += [thisEvalValid[key]]
    # Now that we have collected all the results, we have that each of the four
    # variables (lossTrain, evalTrain, lossValid, evalValid) has a list of lists
    # for each key in the dictionary. The first list goes through the graph, and
    # for each graph, it goes through data realizations. Each data realization
    # is actually an np.array.

    #\\\ COMPUTE STATISTICS:
    # The first thing to do is to transform those into a matrix with all the
    # realizations, so create the variables to save that.
    meanLossTrain = {}
    meanEvalTrain = {}
    meanLossValid = {}
    meanEvalValid = {}
    stdDevLossTrain = {}
    stdDevEvalTrain = {}
    stdDevLossValid = {}
    stdDevEvalValid = {}
    # Initialize the variables
    for thisModel in modelList:
        # Transform into np.array
        lossTrain[thisModel] = np.array(lossTrain[thisModel])
        evalTrain[thisModel] = np.array(evalTrain[thisModel])
        lossValid[thisModel] = np.array(lossValid[thisModel])
        evalValid[thisModel] = np.array(evalValid[thisModel])
        # So, finally, for each model and each graph, we have a np.array of
        # shape:  nDataRealizations x number_of_training_steps
        # And we have to average these to get the mean across all data
        # realizations for each graph
        # And compute the statistics
        meanLossTrain[thisModel] = \
                            np.mean(lossTrain[thisModel], axis = 0)
        meanEvalTrain[thisModel] = \
                            np.mean(evalTrain[thisModel], axis = 0)
        meanLossValid[thisModel] = \
                            np.mean(lossValid[thisModel], axis = 0)
        meanEvalValid[thisModel] = \
                            np.mean(evalValid[thisModel], axis = 0)
        stdDevLossTrain[thisModel] = \
                            np.std(lossTrain[thisModel], axis = 0)
        stdDevEvalTrain[thisModel] = \
                            np.std(evalTrain[thisModel], axis = 0)
        stdDevLossValid[thisModel] = \
                            np.std(lossValid[thisModel], axis = 0)
        stdDevEvalValid[thisModel] = \
                            np.std(evalValid[thisModel], axis = 0)

    ####################
    # SAVE FIGURE DATA #
    ####################

    # And finally, we can plot. But before, let's save the variables mean and
    # stdDev so, if we don't like the plot, we can re-open them, and re-plot
    # them, a piacere.
    #   Pickle, first:
    varsPickle = {}
    varsPickle['nEpochs'] = nEpochs
    varsPickle['nBatches'] = nBatches
    varsPickle['meanLossTrain'] = meanLossTrain
    varsPickle['stdDevLossTrain'] = stdDevLossTrain
    varsPickle['meanEvalTrain'] = meanEvalTrain
    varsPickle['stdDevEvalTrain'] = stdDevEvalTrain
    varsPickle['meanLossValid'] = meanLossValid
    varsPickle['stdDevLossValid'] = stdDevLossValid
    varsPickle['meanEvalValid'] = meanEvalValid
    varsPickle['stdDevEvalValid'] = stdDevEvalValid
    with open(os.path.join(saveDirFigs,'figVars.pkl'), 'wb') as figVarsFile:
        pickle.dump(varsPickle, figVarsFile)
    #   Matlab, second:
    varsMatlab = {}
    varsMatlab['nEpochs'] = nEpochs
    varsMatlab['nBatches'] = nBatches
    for thisModel in modelList:
        varsMatlab['meanLossTrain' + thisModel] = meanLossTrain[thisModel]
        varsMatlab['stdDevLossTrain' + thisModel] = stdDevLossTrain[thisModel]
        varsMatlab['meanEvalTrain' + thisModel] = meanEvalTrain[thisModel]
        varsMatlab['stdDevEvalTrain' + thisModel] = stdDevEvalTrain[thisModel]
        varsMatlab['meanLossValid' + thisModel] = meanLossValid[thisModel]
        varsMatlab['stdDevLossValid' + thisModel] = stdDevLossValid[thisModel]
        varsMatlab['meanEvalValid' + thisModel] = meanEvalValid[thisModel]
        varsMatlab['stdDevEvalValid' + thisModel] = stdDevEvalValid[thisModel]
    savemat(os.path.join(saveDirFigs, 'figVars.mat'), varsMatlab)

    ########
    # PLOT #
    ########

    # Compute the x-axis
    xTrain = np.arange(0, nEpochs * nBatches, xAxisMultiplierTrain)
    xValid = np.arange(0, nEpochs * nBatches, \
                          validationInterval*xAxisMultiplierValid)

    # If we do not want to plot all the elements (to avoid overcrowded plots)
    # we need to recompute the x axis and take those elements corresponding
    # to the training steps we want to plot
    if xAxisMultiplierTrain > 1:
        # Actual selected samples
        selectSamplesTrain = xTrain
        # Go and fetch tem
        for thisModel in modelList:
            meanLossTrain[thisModel] = meanLossTrain[thisModel]\
                                                    [selectSamplesTrain]
            stdDevLossTrain[thisModel] = stdDevLossTrain[thisModel]\
                                                        [selectSamplesTrain]
            meanEvalTrain[thisModel] = meanEvalTrain[thisModel]\
                                                    [selectSamplesTrain]
            stdDevEvalTrain[thisModel] = stdDevEvalTrain[thisModel]\
                                                        [selectSamplesTrain]
    # And same for the validation, if necessary.
    if xAxisMultiplierValid > 1:
        selectSamplesValid = np.arange(0, len(meanLossValid[thisModel]), \
                                       xAxisMultiplierValid)
        for thisModel in modelList:
            meanLossValid[thisModel] = meanLossValid[thisModel]\
                                                    [selectSamplesValid]
            stdDevLossValid[thisModel] = stdDevLossValid[thisModel]\
                                                        [selectSamplesValid]
            meanEvalValid[thisModel] = meanEvalValid[thisModel]\
                                                    [selectSamplesValid]
            stdDevEvalValid[thisModel] = stdDevEvalValid[thisModel]\
                                                        [selectSamplesValid]

    #\\\ LOSS (Training and validation) for EACH MODEL
    for key in meanLossTrain.keys():
        lossFig = plt.figure(figsize=(1.61*5, 1*5))
        plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                     color = '#01256E', linewidth = 2,
                     marker = 'o', markersize = 3)
        plt.errorbar(xValid, meanLossValid[key], yerr = stdDevLossValid[key],
                     color = '#95001A', linewidth = 2,
                     marker = 'o', markersize = 3)
        plt.ylabel(r'Loss')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        lossFig.savefig(os.path.join(saveDirFigs,'loss%s.pdf' % key),
                        bbox_inches = 'tight')

    #\\\ ACCURACY (Training and validation) for EACH MODEL
    for key in meanEvalTrain.keys():
        accFig = plt.figure(figsize=(1.61*5, 1*5))
        plt.errorbar(xTrain, meanEvalTrain[key], yerr = stdDevEvalTrain[key],
                     color = '#01256E', linewidth = 2,
                     marker = 'o', markersize = 3)
        plt.errorbar(xValid, meanEvalValid[key], yerr = stdDevEvalValid[key],
                     color = '#95001A', linewidth = 2,
                     marker = 'o', markersize = 3)
        plt.ylabel(r'Accuracy')
        plt.xlabel(r'Training steps')
        plt.legend([r'Training', r'Validation'])
        plt.title(r'%s' % key)
        accFig.savefig(os.path.join(saveDirFigs,'eval%s.pdf' % key),
                        bbox_inches = 'tight')

    # LOSS (training) for ALL MODELS
    allLossTrain = plt.figure(figsize=(1.61*5, 1*5))
    for key in meanLossTrain.keys():
        plt.errorbar(xTrain, meanLossTrain[key], yerr = stdDevLossTrain[key],
                     linewidth = 2, marker = 'o', markersize = 3)
    plt.ylabel(r'Loss')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanLossTrain.keys()))
    allLossTrain.savefig(os.path.join(saveDirFigs,'allLossTrain.pdf'),
                    bbox_inches = 'tight')

    # ACCURACY (validation) for ALL MODELS
    allEvalValid = plt.figure(figsize=(1.61*5, 1*5))
    for key in meanEvalValid.keys():
        plt.errorbar(xValid, meanEvalValid[key], yerr = stdDevEvalValid[key],
                     linewidth = 2, marker = 'o', markersize = 3)
    plt.ylabel(r'Accuracy')
    plt.xlabel(r'Training steps')
    plt.legend(list(meanEvalValid.keys()))
    allEvalValid.savefig(os.path.join(saveDirFigs,'allEvalValid.pdf'),
                    bbox_inches = 'tight')
