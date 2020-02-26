# 2018/12/4~
"""
dataTools.py Data management module

Several tools to manage data

SourceLocalization (class): creates the datasets for a source localization 
    problem
Authorship (class): loads and splits the dataset for the authorship attribution
    problem
TwentyNews (class): handles the 20NEWS dataset
"""

import numpy as np
import torch
import pickle as pkl
#import hdf5storage # This is required to import old Matlab(R) files.

import Utils.graphTools as graph
import Utils.miscTools as misc

class _dataForClassification:
    # Internal supraclass from which data classes inherit.
    # There are certian methods that all Data classes must have:
    #   getSamples(), evluate(), to() and astype().
    # To avoid coding this methods over and over again, we create a class from
    # which the data can inherit this basic methods.
    # Note that this is called "ForClassification" since the evaluate method
    # is only for classification evaluations.
    # However, it is true that getSamples() might be useful beyond the 
    # classification problem, so we might, eventually, consider a different
    # internal class.
    def __init__(self):
        # Minimal set of attributes that all data classes should have
        self.dataType = None
        self.device = None
        self.nTrain = None
        self.nValid = None
        self.nTest = None
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = None
        self.samples['train']['labels'] = None
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = None
        self.samples['valid']['labels'] = None
        self.samples['test'] = {}
        self.samples['test']['signals'] = None
        self.samples['test']['labels'] = None
        
    def getSamples(self, samplesType, *args):
        # type: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['labels']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # The reshape is to avoid squeezing if only one sample is
                # requested
                x = x[selectedIndices,:].reshape([args[0], x.shape[1]])
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                x = x[args[0], :]
                # If only one element is selected, avoid squeezing. Given that
                # the element can be a list (which has property len) or an
                # np.array (which doesn't have len, but shape), then we can
                # only avoid squeezing if we check that it has been sequeezed
                # (or not)
                if len(x.shape) == 1:
                    x = x.reshape([1, x.shape[0]])
                # And assign the labels
                y = y[args[0]]

        return x, y

    def astype(self, dataType):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        if repr(dataType).find('torch') == -1:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                       = dataType(self.samples[key][secondKey])
        else:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                    = torch.tensor(self.samples[key][secondKey]).type(dataType)

        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This changes the type for the minimal attributes (samples). This 
        # methods should still be initialized within the data classes, if more
        # attributes are used.
        # This can only be done if they are torch tensors
        if repr(self.dataType).find('torch') >= 0:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                      = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device

    def evaluate(self, yHat, y, tol = 1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """
        N = len(y)
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
            yHat = torch.argmax(yHat, dim = 1).type(self.dataType)
            #   And compute the error
            totalErrors = torch.sum(torch.abs(yHat - y) > tol)
            accuracy = 1 - totalErrors.type(self.dataType)/N
        else:
            yHat = np.array(yHat)
            y = np.array(y)
            #   We compute the target label (hardmax)
            yHat = np.argmax(yHat, axis = 1).astype(y.dtype)
            #   And compute the error
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            accuracy = 1 - totalErrors.astype(self.dataType)/N
        #   And from that, compute the accuracy
        return accuracy
        

class SourceLocalization(_dataForClassification):
    """
    SourceLocalization: Creates the dataset for a source localization problem

    Initialization:

    Input:
        G (class): Graph on which to diffuse the process, needs an attribute
            .N with the number of nodes (int) and attribute .W with the
            adjacency matrix (np.array)
        nTrain (int): number of training samples
        nValid (int): number of validation samples
        nTest (int): number of testing samples
        sourceNodes (list of int): list of indices of nodes to be used as
            sources of the diffusion process
        tMax (int): maximum diffusion time, if None, the maximum diffusion time
            is the size of the graph (default: None)
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved.

    Methods:

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """

    def __init__(self, G, nTrain, nValid, nTest, sourceNodes, tMax = None,
                 dataType = np.float64, device = 'cpu'):
        # Initialize parent
        super().__init__()
        # store attributes
        self.dataType = dataType
        self.device = device
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        # If no tMax is specified, set it the maximum possible.
        if tMax == None:
            tMax = G.N
        #\\\ Generate the samples
        # Get the largest eigenvalue of the weighted adjacency matrix
        EW, VW = graph.computeGFT(G.W, order = 'totalVariation')
        eMax = np.max(EW)
        # Normalize the matrix so that it doesn't explode
        Wnorm = G.W / eMax
        # total number of samples
        nTotal = nTrain + nValid + nTest
        # sample source nodes
        sampledSources = np.random.choice(sourceNodes, size = nTotal)
        # sample diffusion times
        sampledTimes = np.random.choice(tMax, size = nTotal)
        # Since the signals are generated as W^t * delta, this reduces to the
        # selection of a column of W^t (the column corresponding to the source
        # node). Therefore, we generate an array of size tMax x N x N with all
        # the powers of the matrix, and then we just simply select the
        # corresponding column for the corresponding time
        lastWt = np.eye(G.N, G.N)
        Wt = lastWt.reshape([1, G.N, G.N])
        for t in range(1,tMax):
            lastWt = lastWt @ Wnorm
            Wt = np.concatenate((Wt, lastWt.reshape([1, G.N, G.N])), axis = 0)
        x = Wt[sampledTimes, :, sampledSources]
        # Now, we have the signals and the labels
        signals = x # nTotal x N (CS notation)
        # Finally, we have to match the source nodes to the corresponding labels
        # which start at 0 and increase in integers.
        nodesToLabels = {}
        for it in range(len(sourceNodes)):
            nodesToLabels[sourceNodes[it]] = it
        labels = [nodesToLabels[x] for x in sampledSources] # nTotal
        # Split and save them
        self.samples['train']['signals'] = signals[0:nTrain, :]
        self.samples['train']['labels'] = labels[0:nTrain]
        self.samples['valid']['signals'] = signals[nTrain:nTrain+nValid, :]
        self.samples['valid']['labels'] = labels[nTrain:nTrain+nValid]
        self.samples['test']['signals'] = signals[nTrain+nValid:nTotal, :]
        self.samples['test']['labels'] = labels[nTrain+nValid:nTotal]
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
    
class Authorship(_dataForClassification):
    """
    Authorship: Loads the dataset of 19th century writers for the authorship
        attribution problem

    Initialization:

    Input:
        authorName (string): which is the selected author to attribute plays to
        ratioTrain (float): ratio of the total texts to be part of the training
            set
        ratioValid (float): ratio of the train texts to be part of the
            validation set
        dataPath (string): path to where the authorship data is located
        graphNormalizationType ('rows' or 'cols'): how to normalize the created
            graph from combining all the selected author WANs
        keepIsolatedNodes (bool): If False, get rid of isolated nodes
        forceUndirected (bool): If True, create an undirected graph
        forceConnected (bool): If True, ensure that the resulting graph is
            connected
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:
        
    .loadData(dataPath): load the data found in dataPath and store it in 
        attributes .authorData and .functionWords
        
    authorData = .getAuthorData(samplesType, selectData, [, optionalArguments])
    
        Input:
            samplesType (string): 'train', 'valid', 'test' or 'all' to determine
                from which dataset to get the raw author data from
            selectData (string): 'WAN' or 'wordFreq' to decide if we want to
                retrieve either the WAN of each excerpt or the word frequency
                count of each excerpt
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        
        Output:
            Either the WANs or the word frequency count of all the excerpts of
            the selected author
            
    .createGraph(): creates a graph from the WANs of the excerpt written by the
        selected author available in the training set. The fusion of this WANs
        is done in accordance with the input options following 
        graphTools.createGraph().
        The resulting adjacency matrix is stored.
        
    .getGraph(): fetches the stored adjacency matrix and returns it
    
    .getFunctionWords(): fetches the list of functional words. Returns a tuple
        where the first element correspond to all the functional words in use, 
        and the second element consists of all the functional words available.
        Obs.: When we created the graph, some of the functional words might have
        been dropped in order to make it connected, for example.

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """
    
    def __init__(self, authorName, ratioTrain, ratioValid, dataPath,
                 graphNormalizationType, keepIsolatedNodes,
                 forceUndirected, forceConnected,
                 dataType = np.float64, device = 'cpu'):
        # Initialize parent
        super().__init__()
        # Store
        self.authorName = authorName
        self.dataPath = dataPath
        self.dataType = dataType
        self.device = device
        # Store characteristics of the graph to be created
        self.graphNormalizationType = graphNormalizationType
        self.keepIsolatedNodes = keepIsolatedNodes
        self.forceUndirected = forceUndirected
        self.forceConnected = forceConnected
        self.adjacencyMatrix = None
        # Other data to save
        self.authorData = None
        self.selectedAuthor = None
        self.allFunctionWords = None
        self.functionWords = None
        # Load data
        self.loadData(dataPath)
        # Check that the authorName is a valid name
        assert authorName in self.authorData.keys()
        # Get the selected author's data
        thisAuthorData = self.authorData[authorName].copy()
        nExcerpts = thisAuthorData['wordFreq'].shape[0] # Number of excerpts
            # by the selected author
        nTrainAuthor = round(ratioTrain * nExcerpts)
        nValidAuthor = round(ratioValid * nTrainAuthor)
        nTestAuthor = nExcerpts - nTrainAuthor
        nTrainAuthor = nTrainAuthor - nValidAuthor
        # Now, we know how many training, validation and testing samples from
        # the required author. But we will also include an equal amount of
        # other authors, therefore
        self.nTrain = round(2 * nTrainAuthor)
        self.nValid = round(2 * nValidAuthor)
        self.nTest = round(2 * nTestAuthor)
        
        # Now, let's get the corresponding signals for the author
        xAuthor = thisAuthorData['wordFreq']
        # Get a random permutation of these works, and split them accordingly
        randPerm = np.random.permutation(nExcerpts)
        # Save the indices corresponding to each split
        randPermTrain = randPerm[0:nTrainAuthor]
        randPermValid = randPerm[nTrainAuthor:nTrainAuthor+nValidAuthor]
        randPermTest = randPerm[nTrainAuthor+nValidAuthor:nExcerpts]
        xAuthorTrain = xAuthor[randPermTrain, :]
        xAuthorValid = xAuthor[randPermValid, :]
        xAuthorTest = xAuthor[randPermTest, :]
        # And we will store this split
        self.selectedAuthor = {}
        # Copy all data
        self.selectedAuthor['all'] = thisAuthorData.copy()
        # Copy word frequencies
        self.selectedAuthor['train'] = {}
        self.selectedAuthor['train']['wordFreq'] = xAuthorTrain.copy()
        self.selectedAuthor['valid'] = {}
        self.selectedAuthor['valid']['wordFreq'] = xAuthorValid.copy()
        self.selectedAuthor['test'] = {}
        self.selectedAuthor['test']['wordFreq'] = xAuthorTest.copy()
        # Copy WANs
        self.selectedAuthor['train']['WAN'] = \
                              thisAuthorData['WAN'][randPermTrain, :, :].copy()
        self.selectedAuthor['valid']['WAN'] = \
                              thisAuthorData['WAN'][randPermValid, :, :].copy()
        self.selectedAuthor['test']['WAN'] = \
                               thisAuthorData['WAN'][randPermTest, :, :].copy()
        # Now we need to get an equal amount of works from the rest of the
        # authors.
        xRest = np.empty([0, xAuthorTrain.shape[1]]) # Create an empty matrix
        # to store all the works by the rest of the authors.
        # Now go author by author gathering all works
        for key in self.authorData.keys():
            # Only for authors that are not the selected author
            if key is not authorName:
                thisAuthorTexts = self.authorData[key]['wordFreq']
                xRest = np.concatenate((xRest, thisAuthorTexts), axis = 0)
        # After obtaining all works, xRest is of shape nRestOfData x nWords
        # We now need to select at random from this other data, but only up
        # to nExcerpts. Therefore, we will randperm all the indices, but keep
        # only the first nExcerpts indices.
        randPerm = np.random.permutation(xRest.shape[0])
        randPerm = randPerm[0:nExcerpts] # nExcerpts x nWords
        # And now we should just get the appropriate number of texts from these
        # other authors.
        # Compute how many samples for each case
        nTrainRest = self.nTrain - nTrainAuthor
        nValidRest = self.nValid - nValidAuthor
        nTestRest = self.nTest - nTestAuthor
        # And obtain those
        xRestTrain = xRest[randPerm[0:nTrainRest], :]
        xRestValid = xRest[randPerm[nTrainRest:nTrainRest + nValidRest], :]
        xRestTest = xRest[randPerm[nTrainRest+nValidRest:nExcerpts], :]
        # Now construct the signals and labels. Signals is just the 
        # concatenation of each of these excerpts. Labels is just a bunch of
        # 1s followed by a bunch of 0s
        # Obs.: The fact that the dataset is ordered now, it doesn't matter,
        # since it will be shuffled at each epoch.
        xTrain = np.concatenate((xAuthorTrain, xRestTrain), axis = 0)
        labelsTrain = np.concatenate((np.ones(nTrainAuthor),
                                      np.zeros(nTrainRest)), axis = 0)
        xValid = np.concatenate((xAuthorValid, xRestValid), axis = 0)
        labelsValid = np.concatenate((np.ones(nValidAuthor),
                                      np.zeros(nValidRest)), axis = 0)
        xTest = np.concatenate((xAuthorTest, xRestTest), axis = 0)
        labelsTest = np.concatenate((np.ones(nTestAuthor),
                                     np.zeros(nTestRest)), axis = 0)
        # And assign them to the required attribute samples
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = xTrain
        self.samples['train']['labels'] = labelsTrain
        self.samples['valid'] = {}
        self.samples['valid']['signals'] = xValid
        self.samples['valid']['labels'] = labelsValid
        self.samples['test'] = {}
        self.samples['test']['signals'] = xTest
        self.samples['test']['labels'] = labelsTest
        # Create graph
        self.createGraph()
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
    
    def loadData(self, dataPath):
        # TODO: Analyze if it's worth it to create a .pkl and load that 
        # directly once the data has been appropriately parsed. It's just
        # that loading with hdf5storage takes a couple of second that
        # could be saved if the .pkl file is faster.
        rawData = hdf5storage.loadmat(dataPath)
        # rawData is a dictionary with four keys:
        #   'all_authors': contains the author list
        #   'all_freqs': contains the word frequency count for each excerpt
        #   'all_wans': contains the WANS for each excerpt
        #   'function_words': a list of the functional words
        # The issue is that hdf5storage, while necessary to load old 
        # Matlab(R) files, gives the data in a weird format, that we need
        # to adapt and convert.
        # The data will be structured as follows. We will have an
        # authorData dictionary of dictionaries: the first key will be the
        # author name, the second key will be either freqs or wans to
        # access either one or another.
        # We will also clean up and save the functional word list, although
        # we do not need to use it.
        authorData = {} # Create dictionary
        for it in range(len(rawData['all_authors'])):
            thisAuthor = str(rawData['all_authors'][it][0][0][0])
            # Each element in rawData['all_authors'] is nested in a couple
            # of lists, so that's why we need the three indices [0][0][0] 
            # to reach the string with the actual author name.
            # Get the word frequency
            thisWordFreq = rawData['all_freqs'][0][it] # 1 x nWords x nData
            # Again, the [0] is due to the structure of the data
            # Let us get rid of that extra 1, and then transpose this to be
            # stored as nData x nWords (since nWords is the dimension of 
            # the number of nodes the network will have; CS notation)
            thisWordFreq = thisWordFreq.squeeze(0).T # nData x nWords
            # Finally, get the WANs
            thisWAN = rawData['all_wans'][0][it] # nWords x nWords x nData
            thisWAN = thisWAN.transpose(2, 0, 1) # nData x nWords x nWords
            # Obs.: thisWAN is likely not symmetric, so the way this is 
            # transposed matters. In this case, since thisWAN was intended
            # to be a tensor in matlab (where the last index is the 
            # collection of matrices), we just throw that last dimension to
            # the front (since numpy consider the first index as the 
            # collection index).
            # Now we can create the dictionary and save the corresopnding
            # data.
            authorData[thisAuthor] = {}
            authorData[thisAuthor]['wordFreq'] = thisWordFreq
            authorData[thisAuthor]['WAN'] = thisWAN
        # And at last, gather the list of functional words
        functionWords = [] # empty list to store the functional words
        for word in rawData['function_words']:
            functionWords.append(str(word[0][0][0]))
        # Store all the data recently collected
        self.authorData = authorData
        self.allFunctionWords = functionWords
        self.functionWords = functionWords.copy()
        
    def getAuthorData(self, samplesType, dataType, *args):
        # type: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test' or samplesType == 'all'
        # Check that the dataType is either wordFreq or WAN
        assert dataType == 'WAN' or dataType == 'wordFreq'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.selectedAuthor[samplesType][dataType]
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # The reshape is to avoid squeezing if only one sample is
                # requested (because x can have two or three dimension, we
                # need to take a longer path here, so we will only do it
                # if args[0] is equal to 1.)
                if args[0] == 1:
                    newShape = [1]
                    newShape.extend(list(x.shape[1:]))
                    x = x[selectedIndices].reshape(newShape)
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                xNew = x[args[0]]
                # If only one element is selected, avoid squeezing. Given that
                # the element can be a list (which has property len) or an
                # np.array (which doesn't have len, but shape), then we can
                # only avoid squeezing if we check that it has been sequeezed
                # (or not)
                if len(xNew.shape) <= len(x.shape):
                    newShape = [1]
                    newShape.extend(list(x.shape[1:]))
                    x = xNew.reshape(newShape)

        return x
    
    def createGraph(self):
        
        # Save list of nodes to keep to later update the datasets with the
        # appropriate words
        nodesToKeep = []
        # Number of nodes (so far) = Number of functional words
        N = self.selectedAuthor['all']['wordFreq'].shape[1]
        # Create graph
        W = graph.createGraph('fuseEdges', N, self.selectedAuthor['train']['WAN'],
                              'sum', self.graphNormalizationType,
                              self.keepIsolatedNodes, self.forceUndirected,
                              self.forceConnected, nodesToKeep)
        # Store adjacency matrix
        self.adjacencyMatrix = W.astype(np.float64)
        # Update data
        if self.samples['train']['signals'] is not None:
            self.samples['train']['signals'] = \
                                self.samples['train']['signals'][:, nodesToKeep]
        if self.samples['valid']['signals'] is not None:
            self.samples['valid']['signals'] = \
                                self.samples['valid']['signals'][:, nodesToKeep]
        if self.samples['test']['signals'] is not None:
            self.samples['test']['signals'] = \
                                self.samples['test']['signals'][:, nodesToKeep]
        if self.allFunctionWords is not None:
            self.functionWords = [self.allFunctionWords[w] for w in nodesToKeep]
        
    def getGraph(self):
        
        return self.adjacencyMatrix
    
    def getFunctionWords(self):
        
        return self.functionWords, self.allFunctionWords
    
    def astype(self, dataType):
        # This changes the type for the selected author as well as the samples
        # First, the selected author info
        if repr(dataType).find('torch') == -1:
            for key in self.selectedAuthor.keys():
                for secondKey in self.selectedAuthor[key].keys():
                    self.selectedAuthor[key][secondKey] \
                                 = dataType(self.selectedAuthor[key][secondKey])
            self.adjacencyMatrix = dataType(self.adjacencyMatrix)
        else:
            for key in self.selectedAuthor.keys():
                for secondKey in self.selectedAuthor[key].keys():
                    self.selectedAuthor[key][secondKey] \
                            = torch.tensor(self.selectedAuthor[key][secondKey])\
                                    .type(dataType)
            self.adjacencyMatrix = torch.tensor(self.adjacencyMatrix)\
                                        .type(dataType)

        # And now, initialize to change the samples as well (and also save the 
        # data type)
        super().astype(dataType)
        
    
    def to(self, device):
        # If the dataType is 'torch'
        if repr(self.dataType).find('torch') >= 0:
            # Change the selected author ('test', 'train', 'valid', 'all';
            # 'WANs', 'wordFreq')
            for key in self.selectedAuthor.keys():
                for secondKey in self.selectedAuthor[key].keys():
                    self.selectedAuthor[key][secondKey] \
                                = self.selectedAuthor[key][secondKey].to(device)
            self.adjacencyMatrix.to(device)                 
            # And call the inherit method to initialize samples (and save to
            # device)
            super().to(device)
            
class TwentyNews(_dataForClassification):
    """
    TwentyNews: Loads and handles handles the 20NEWS dataset

    Initialization:

    Input:
        ratioValid (float): ratio of the train texts to be part of the
            validation set
        nWords (int): number of words to consider (i.e. the nWords most frequent
            words in the news articles are kept, the rest, discarded)
        nWordsShortDocs (int): any article with less words than nWordsShortDocs
            are discarded.
        nEdges (int): how many edges to keep after creating a geometric graph
            considering the graph embedding of each new article.
        distMetric (string): function to use to compute the distance between
            articles in the embedded space.
        dataDir (string): directory where to download the 20News dataset to/
            to check if it has already been downloaded
        dataType (dtype): type of loaded data (default: np.float64)
        device (device): where to store the data (e.g., 'cpu', 'cuda:0', etc.)

    Methods:
        
    .getData(dataSubset): loads the data belonging to dataSubset (i.e. 'train' 
        or 'test')
    
    .embedData(): compute the graph embedding of the training dataset after
        it has been loaded
    
    .normalizeData(normType): normalize the data in the embedded space following
        a normType norm.
    
    .createValidationSet(ratio): stores ratio% of the training set as validation
        set.
        
    .createGraph(): uses the word2vec embedding of the training set to compute
        a geometric graph
        
    .getGraph(): fetches the adjacency matrix of the stored graph
    
    .getNumberOfClasses(): fetches the number of classes
    
    .reduceDataset(nTrain, nValid, nTest): reduces the dataset by randomly
        selected nTrain, nValid and nTest samples from the training, validation
        and testing datasets, respectively.
        
    authorData = .getAuthorData(samplesType, selectData, [, optionalArguments])
    
        Input:
            samplesType (string): 'train', 'valid', 'test' or 'all' to determine
                from which dataset to get the raw author data from
            selectData (string): 'WAN' or 'wordFreq' to decide if we want to
                retrieve either the WAN of each excerpt or the word frequency
                count of each excerpt
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        
        Output:
            Either the WANs or the word frequency count of all the excerpts of
            the selected author
            
    .createGraph(): creates a graph from the WANs of the excerpt written by the
        selected author available in the training set. The fusion of this WANs
        is done in accordance with the input options following 
        graphTools.createGraph().
        The resulting adjacency matrix is stored.
        
    .getGraph(): fetches the stored adjacency matrix and returns it
    
    .getFunctionWords(): fetches the list of functional words. Returns a tuple
        where the first element correspond to all the functional words in use, 
        and the second element consists of all the functional words available.
        Obs.: When we created the graph, some of the functional words might have
        been dropped in order to make it connected, for example.

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """
    def __init__(self, ratioValid, nWords, nWordsShortDocs, nEdges, distMetric,
                 dataDir, dataType = np.float64, device = 'cpu'):
        
        super().__init__()
        # This creates the attributes: dataType, device, nTrain, nTest, nValid,
        # and samples, and fills them all with None, and also creates the 
        # methods: getSamples, astype, to, and evaluate.
        self.dataType = dataType
        self.device = device
        
        # Other relevant information we need to store:
        self.dataDir = dataDir # Where the data is
        self.N = nWords # Number of nodes
        self.nWordsShortDocs = nWordsShortDocs # Number of words under which
            # a document is too short to be taken into consideration
        self.M = nEdges # Number of edges
        self.distMetric = distMetric # Distance metric to use
        self.dataset = {} # Here we save the dataset classes as they are
            # handled by mdeff's code
        self.nClasses = None # Number of classes
        self.vocab = None # Words considered
        self.graphData = None # Store the data (word2vec embeddings) required
            # to build the graph
        self.adjacencyMatrix = None # Store the graph built from the loaded
            # data
    
        # Get the training dataset. Saves vocab, dataset, and samples
        self.getData('train')
        # Embeds the data following the N words and a word2vec approach, saves
        # the embedded vectors in graphData, and updates vocab to keep only
        # the N words selected
        self.embedData()
        # Get the testing dataset, only for the words stored in vocab.
        self.getData('test')
        # Normalize
        self.normalizeData()
        # Save number of samples
        self.nTrain = self.samples['train']['labels'].shape[0]
        self.nTest = self.samples['test']['labels'].shape[0]
        # Create validation set
        self.createValidationSet(ratioValid)
        # Create graph
        self.createGraph() # Only after data has been embedded
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)
        
    def getData(self, dataSubset):
        
        # Load dataset
        dataset = Text20News(data_home = self.dataDir,
                             subset = dataSubset,
                             remove = ('headers','footers','quotes'),
                             shuffle = True)
        # Get rid of numbers and other stuff
        dataset.clean_text(num='substitute')
        # If there's some vocabulary already defined, vectorize (count the
        # frequencies) of the words in vocab, if not, count all of them
        if self.vocab is None:
            dataset.vectorize(stop_words='english')
            self.vocab = dataset.vocab
        else:
            dataset.vectorize(vocabulary = self.vocab)
    
        # Get rid of short documents
        if dataSubset == 'train':
            dataset.remove_short_documents(nwords = self.nWordsShortDocs,
                                           vocab = 'full')
            # Get rid of images
            dataset.remove_encoded_images()
            self.nClasses = len(dataset.class_names)
        else:
            dataset.remove_short_documents(nwords = self.nWordsShortDocs,
                                           vocab = 'selected')
        
        # Save them in the corresponding places
        self.samples[dataSubset]['signals'] = dataset.data.toarray()
        self.samples[dataSubset]['labels'] = dataset.labels
        self.dataset[dataSubset] = dataset
        
    def embedData(self):
        
        # We need to have loaded the training dataset first.
        assert 'train' in self.dataset.keys()
        # Embed them (word2vec embedding)
        self.dataset['train'].embed()
        # Keep only the top words (which determine the number of nodes)
        self.dataset['train'].keep_top_words(self.N)
        # Update the vocabulary
        self.vocab = self.dataset['train'].vocab
        # Get rid of short documents when considering only the specific 
        # vocabulary
        self.dataset['train'].remove_short_documents(
                                                  nwords = self.nWordsShortDocs,
                                                  vocab = 'selected')
        # Save the embeddings, which are necessary to build a graph
        self.graphData = self.dataset['train'].embeddings
        # Update the samples
        self.samples['train']['signals'] = self.dataset['train'].data.toarray()
        self.samples['train']['labels'] = self.dataset['train'].labels
        # If there's an existing dataset, update it to the new vocabulary
        if 'test' in self.dataset.keys():
            self.dataset['test'].vectorize(vocabulary = self.vocab)
            # Update the samples
            self.samples['test']['signals'] =self.dataset['test'].data.toarray()
            self.samples['test']['labels'] = self.dataset['test'].labels
        
    def normalizeData(self, normType = 'l1'):
        
        for key in self.dataset.keys():
            # Normalize the frequencies on the l1 norm.
            self.dataset[key].normalize(norm = normType)
            # And save it
            self.samples[key]['signals'] = self.dataset[key].data.toarray()
            self.samples[key]['labels'] = self.dataset[key].labels
            
    def createValidationSet(self, ratio):
        # How many valid samples
        self.nValid = int(ratio * self.nTrain)
        # Shuffle indices
        randomIndices = np.random.permutation(self.nTrain)
        validationIndices = randomIndices[0:self.nValid]
        trainIndices = randomIndices[self.nValid:]
        # Fetch those samples and put them in the validation set
        self.samples['valid']['signals'] = self.samples['train']['signals']\
                                                          [validationIndices, :]
        self.samples['valid']['labels'] = self.samples['train']['labels']\
                                                             [validationIndices]
        # And update the training set
        self.samples['train']['signals'] = self.samples['train']['signals']\
                                                               [trainIndices, :]
        self.samples['train']['labels'] = self.samples['train']['labels']\
                                                                  [trainIndices]
        # Update the numbers
        self.nValid = self.samples['valid']['labels'].shape[0]
        self.nTrain = self.samples['train']['labels'].shape[0]
            
    def createGraph(self, *args):
        
        assert self.graphData is not None
        assert len(args) == 0 or len(args) == 2
        if len(args) == 2:
            self.M = args[0] # Number of edges
            self.distMetric = args[1] # Distance metric
        dist, idx = distance_sklearn_metrics(self.graphData, k = self.M,
                                             metric = self.distMetric)
        self.adjacencyMatrix = adjacency(dist, idx).toarray()
        
    def getGraph(self):
        
        return self.adjacencyMatrix
    
    def getNumberOfClasses(self):
        
        return self.nClasses
    
    def reduceDataset(self, nTrain, nValid, nTest):
        if nTrain < self.nTrain:
            randomIndices = np.random.permutation(self.nTrain)
            trainIndices = randomIndices[0:nTrain]
            # And update the training set
            self.samples['train']['signals'] = self.samples['train']\
                                                           ['signals']\
                                                           [trainIndices, :]
            self.samples['train']['labels'] = self.samples['train']\
                                                          ['labels']\
                                                          [trainIndices]
            self.nTrain = nTrain
        if nValid < self.nValid:
            randomIndices = np.random.permutation(self.nValid)
            validIndices = randomIndices[0:nValid]
            # And update the training set
            self.samples['valid']['signals'] = self.samples['valid']\
                                                           ['signals']\
                                                           [validIndices, :]
            self.samples['valid']['labels'] = self.samples['valid']\
                                                          ['labels']\
                                                          [validIndices]
            self.nValid = nValid
        if nTest < self.nTest:
            randomIndices = np.random.permutation(self.nTest)
            testIndices = randomIndices[0:nTest]
            # And update the training set
            self.samples['test']['signals'] = self.samples['test']\
                                                           ['signals']\
                                                           [testIndices, :]
            self.samples['test']['labels'] = self.samples['test']\
                                                          ['labels']\
                                                          [testIndices]
            self.nTest = nTest
    
    def astype(self, dataType):
        # This changes the type for the graph data, as well as the adjacency
        # matrix. We are going to leave the dataset attribute as it is, since
        # this is the most accurate reflection of mdeff's code.
        if repr(dataType).find('torch') == -1:
            self.graphData = dataType(self.graphData)
            self.adjacencyMatrix = dataType(self.adjacencyMatrix)
        else:
            self.graphData = torch.tensor(self.graphData).type(dataType)
            self.adjacencyMatrix = torch.tensor(self.adjacencyMatrix)\
                                        .type(dataType)

        # And now, initialize to change the samples as well (and also save the 
        # data type)
        super().astype(dataType)
        
    
    def to(self, device):
        # If the dataType is 'torch'
        if repr(self.dataType).find('torch') >= 0:
            # Change the stored attributes that are not handled by the inherited
            # method to().
            self.graphData.to(device)
            self.adjacencyMatrix.to(device)
            # And call the inherit method to initialize samples (and save to
            # device)
            super().to(device)
    
# Copied almost verbatim from the code by Michäel Defferrard, available at
# http://github.com/mdeff/cnn_graph
       
import gensim
import sklearn, sklearn.datasets, sklearn.metrics
import scipy.sparse

import re

def distance_sklearn_metrics(z, k=4, metric='euclidean'):
	"""Compute exact pairwise distances."""
	d = sklearn.metrics.pairwise.pairwise_distances(
			z, metric=metric)
	# k-NN graph.
	idx = np.argsort(d)[:, 1:k+1]
	d.sort()
	d = d[:, 1:k+1]
	return d, idx

def adjacency(dist, idx):
	"""Return the adjacency matrix of a kNN graph."""
	M, k = dist.shape
	assert M, k == idx.shape
	assert dist.min() >= 0

	# Weights.
	sigma2 = np.mean(dist[:, -1])**2
	dist = np.exp(- dist**2 / sigma2)

	# Weight matrix.
	I = np.arange(0, M).repeat(k)
	J = idx.reshape(M*k)
	V = dist.reshape(M*k)
	W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

	# No self-connections.
	W.setdiag(0)

	# Non-directed graph.
	bigger = W.T > W
	W = W - W.multiply(bigger) + W.T.multiply(bigger)

	assert W.nnz % 2 == 0
	assert np.abs(W - W.T).mean() < 1e-10
	assert type(W) is scipy.sparse.csr.csr_matrix
	return W

def replace_random_edges(A, noise_level):
	"""Replace randomly chosen edges by random edges."""
	M, M = A.shape
	n = int(noise_level * A.nnz // 2)

	indices = np.random.permutation(A.nnz//2)[:n]
	rows = np.random.randint(0, M, n)
	cols = np.random.randint(0, M, n)
	vals = np.random.uniform(0, 1, n)
	assert len(indices) == len(rows) == len(cols) == len(vals)

	A_coo = scipy.sparse.triu(A, format='coo')
	assert A_coo.nnz == A.nnz // 2
	assert A_coo.nnz >= n
	A = A.tolil()

	for idx, row, col, val in zip(indices, rows, cols, vals):
		old_row = A_coo.row[idx]
		old_col = A_coo.col[idx]

		A[old_row, old_col] = 0
		A[old_col, old_row] = 0
		A[row, col] = 1
		A[col, row] = 1

	A.setdiag(0)
	A = A.tocsr()
	A.eliminate_zeros()
	return A
        
class TextDataset(object):
    def clean_text(self, num='substitute'):
        # TODO: stemming, lemmatisation
        for i,doc in enumerate(self.documents):
            # Digits.
            if num is 'spell':
                doc = doc.replace('0', ' zero ')
                doc = doc.replace('1', ' one ')
                doc = doc.replace('2', ' two ')
                doc = doc.replace('3', ' three ')
                doc = doc.replace('4', ' four ')
                doc = doc.replace('5', ' five ')
                doc = doc.replace('6', ' six ')
                doc = doc.replace('7', ' seven ')
                doc = doc.replace('8', ' eight ')
                doc = doc.replace('9', ' nine ')
            elif num is 'substitute':
                # All numbers are equal. Useful for embedding
                # (countable words) ?
                doc = re.sub('(\\d+)', ' NUM ', doc)
            elif num is 'remove':
                # Numbers are uninformative (they are all over the place).
                # Useful for bag-of-words ?
                # But maybe some kind of documents contain more numbers,
                # e.g. finance.
                # Some documents are indeed full of numbers. At least
                # in 20NEWS.
                doc = re.sub('[0-9]', ' ', doc)
            # Remove everything except a-z characters and single space.
            doc = doc.replace('$', ' dollar ')
            doc = doc.lower()
            doc = re.sub('[^a-z]', ' ', doc)
            doc = ' '.join(doc.split()) # same as 
                                        # doc = re.sub('\s{2,}', ' ', doc)
            self.documents[i] = doc

    def vectorize(self, **params):
        # TODO: count or tf-idf. Or in normalize ?
        vectorizer = sklearn.feature_extraction.text.CountVectorizer(**params)
        self.data = vectorizer.fit_transform(self.documents)
        self.vocab = vectorizer.get_feature_names()
        assert len(self.vocab) == self.data.shape[1]

    def keep_documents(self, idx):
        """Keep the documents given by the index, discard the others."""
        self.documents = [self.documents[i] for i in idx]
        self.labels = self.labels[idx]
        self.data = self.data[idx,:]

    def keep_words(self, idx):
        """Keep the documents given by the index, discard the others."""
        self.data = self.data[:,idx]
        self.vocab = [self.vocab[i] for i in idx]
        try:
            self.embeddings = self.embeddings[idx,:]
        except AttributeError:
            pass

    def remove_short_documents(self, nwords, vocab='selected'):
        """Remove a document if it contains less than nwords."""
        if vocab is 'selected':
            # Word count with selected vocabulary.
            wc = self.data.sum(axis=1)
            wc = np.squeeze(np.asarray(wc))
        elif vocab is 'full':
            # Word count with full vocabulary.
            wc = np.empty(len(self.documents), dtype=np.int)
            for i,doc in enumerate(self.documents):
                wc[i] = len(doc.split())
        idx = np.argwhere(wc >= nwords).squeeze()
        self.keep_documents(idx)

    def keep_top_words(self, M):
        """Keep in the vocaluary the M words who appear most often."""
        freq = self.data.sum(axis=0)
        freq = np.squeeze(np.asarray(freq))
        idx = np.argsort(freq)[::-1]
        idx = idx[:M]
        self.keep_words(idx)

    def normalize(self, norm='l1'):
        """Normalize data to unit length."""
        # TODO: TF-IDF.
        data = self.data.astype(np.float64)
        self.data = sklearn.preprocessing.normalize(data, axis=1, norm=norm)

    def embed(self, filename=None, size=100):
        """Embed the vocabulary using pre-trained vectors."""
        if filename:
            model = gensim.models.Word2Vec.load_word2vec_format(filename,
                                                                binary=True)
            size = model.vector_size
        else:
            class Sentences(object):
                def __init__(self, documents):
                    self.documents = documents
                def __iter__(self):
                    for document in self.documents:
                        yield document.split()
            model = gensim.models.Word2Vec(Sentences(self.documents), size=size)
        self.embeddings = np.empty((len(self.vocab), size))
        keep = []
        not_found = 0
        for i,word in enumerate(self.vocab):
            try:
                self.embeddings[i,:] = model[word]
                keep.append(i)
            except KeyError:
                not_found += 1
        self.keep_words(keep)
        
    def remove_encoded_images(self, freq=1e3):
        widx = self.vocab.index('ax')
        wc = self.data[:,widx].toarray().squeeze()
        idx = np.argwhere(wc < freq).squeeze()
        self.keep_documents(idx)

class Text20News(TextDataset):
    def __init__(self, **params):
        dataset = sklearn.datasets.fetch_20newsgroups(**params)
        self.documents = dataset.data
        self.labels = dataset.target
        self.class_names = dataset.target_names
        assert max(self.labels) + 1 == len(self.class_names)
        
class KStepPrediction():
    """
    KStepPrediction: Creates the dataset for a K-step prediction problem

    Initialization:

    Input:
        G (class): Graph on which to diffuse the process, needs an attribute
            .N with the number of nodes (int) and attribute .W with the
            adjacency matrix (np.array)
        nTrain (int): number of training samples
        nValid (int): number of validation samples
        nTest (int): number of testing samples
        horizon (int): length of the process
        sigmaSpatial (float): spatial variance
        sigmaTemporal (float): temporal variance
        rhoSpatial (float): spatial correlation
        rhoTemporal (float): temporal correlation
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved.

    Methods:

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    RMSE = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            RMSE (float): root mean square error between y and yHat

    """

    def __init__(self, K, G, nTrain, nValid, nTest, horizon, sigmaSpatial = 1, sigmaTemporal = 0,
                 rhoSpatial = 0, rhoTemporal = 0, dataType = np.float64, device = 'cpu'):
        # store attributes
        self.dataType = dataType
        self.device = device
        self.K = K
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        self.horizon = horizon
        self.sigmaSpatial = sigmaSpatial
        self.sigmaTemporal = sigmaTemporal
        self.rhoSpatial = rhoSpatial
        self.rhoTemporal = rhoTemporal
        #\\\ Generate the samples
        # Get the largest eigenvalue of the weighted adjacency matrix
        EW, VW = graph.computeGFT(G.W, order = 'totalVariation')
        eMax = np.max(EW)
        # Normalize the matrix so that it doesn't explode
        Wnorm = G.W / eMax
        # total number of samples
        nTotal = nTrain + nValid + nTest
        # x_0
        x_t = np.random.rand(nTotal,G.N);
        x = x_t
        # Temporal noise
        tempNoise = np.random.multivariate_normal(np.zeros(self.horizon),
                                                  np.power(self.sigmaTemporal,2)*np.eye(self.horizon) + 
                                                  np.power(self.rhoTemporal,2)*np.ones((self.horizon,self.horizon)),
                                                  (nTotal, G.N))
        tempNoise = np.transpose(tempNoise, (2,0,1))
        # Create LS
        A = Wnorm # = A x_t + w (Gaussian noise)
        for t in range(self.horizon):
            spatialNoise = np.random.multivariate_normal(np.zeros(G.N), 
                                 np.power(self.sigmaSpatial,2)*np.eye(G.N) + 
                                 np.power(self.rhoSpatial,2)*np.ones((G.N,G.N)), nTotal)

            x_tplus1 = np.matmul(x_t,A) + spatialNoise + tempNoise[t, :, :]
            x_t = x_tplus1
            
            x = np.concatenate((x,x_t),axis=1)
        y = x[:,K*G.N:horizon*G.N]
        x = x[:,0:(horizon*G.N-K*G.N)] 
        # Now, we have the signals and the labels
        signals = x # nTotal x N (CS notation)
        labels = y
        # Split and save them
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = signals[0:nTrain, :]
        self.samples['train']['labels'] = labels[0:nTrain, :]
        self.samples['valid'] = {} 
        self.samples['valid']['signals'] = signals[nTrain:nTrain+nValid, :]
        self.samples['valid']['labels'] = labels[nTrain:nTrain+nValid, :]
        self.samples['test'] = {}
        self.samples['test']['signals'] = signals[nTrain+nValid:nTotal, :]
        self.samples['test']['labels'] = labels[nTrain+nValid:nTotal, :]
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    def getSamples(self, samplesType, *args):
        # type: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['labels']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # The reshape is to avoid squeezing if only one sample is
                # requested
                x = x[selectedIndices,:].reshape([args[0], x.shape[1]])
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                x = x[args[0], :]
                # If only one element is selected, avoid squeezing. Given that
                # the element can be a list (which has property len) or an
                # np.array (which doesn't have len, but shape), then we can
                # only avoid squeezing if we check that it has been sequeezed
                # (or not)
                if len(x.shape) == 1:
                    x = x.reshape([1, x.shape[0]])
                # And assign the labels
                y = y[args[0]]

        return x, y

    def astype(self, dataType):
        if repr(dataType).find('torch') == -1:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                            = dataType(self.samples[key][secondKey])
        else:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                         = torch.tensor(self.samples[key][secondKey]).type(dataType)

        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This can only be done if they are torch tensors
        if repr(self.dataType).find('torch') >= 0:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                       = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device

    def evaluate(self, yHat, y, tol = 1e-9):
        """
        Return the MSE loss
        """
        lossValue = misc.batchTimeMSELoss(yHat,y.type(torch.float64))
        return lossValue
    
class QuakeData():
    """
    QuakeData: Manages the dataset for earthquake epicenter estimation

    Initialization:

    Input:
        nTrain (int): percentage of training samples
        nValid (int): percentage of validation samples
        nTest (int): percentage of testing samples
        seqLen (int): length of the process
        downsamplingFactor (float): downsampling factor
        dataType (dtype): datatype for the samples created (default: np.float64)
        device (device): if torch.Tensor datatype is selected, this is on what
            device the data is saved.

    Methods:

    signals, labels = .getSamples(samplesType[, optionalArguments])
        Input:
            samplesType (string): 'train', 'valid' or 'test' to determine from
                which dataset to get the samples from
            optionalArguments:
                0 optional arguments: get all the samples from the specified set
                1 optional argument (int): number of samples to get (at random)
                1 optional argument (list): specific indices of samples to get
        Output:
            signals (dtype.array): numberSamples x numberNodes
            labels (dtype.array): numberSamples
            >> Obs.: The 0th dimension matches the corresponding signal to its
                respective label

    .astype(type): change the type of the data matrix arrays.
        Input:
            type (dtype): target type of the variables (e.g. torch.float64,
                numpy.float64, etc.)

    .to(device): if dtype is torch.tensor, move them to the specified device.
        Input:
            device (string): target device to move the variables to (e.g. 'cpu',
                'cuda:0', etc.)

    accuracy = .evaluate(yHat, y, tol = 1e-9)
        Input:
            yHat (dtype.array): estimated labels (1-D binary vector)
            y (dtype.array): correct labels (1-D binary vector)
            >> Obs.: both arrays are of the same length
            tol (float): numerical tolerance to consider two numbers to be equal
        Output:
            accuracy (float): proportion of correct labels

    """

    def __init__(self, nTrain, nValid, nTest, seqLen, downsamplingFactor,
                 dataType = np.float64, device = 'cpu'):
        # store attributes
        self.dataType = dataType
        self.device = device
        self.nTrain = nTrain
        self.nValid = nValid
        self.nTest = nTest
        nTotal = nTrain + nValid + nTest
        self.seqLen = seqLen
        self.downSamplingFactor = downsamplingFactor
        #\\\ Generate the samples
        X = pkl.load(open("X.p","rb"))
        y = pkl.load(open("y.p","rb"))
        y = y.astype(int)
        y = y.reshape(-1,1) #        onehot_encoder = OneHotEncoder(n_values=59,sparse=False)
#        y = onehot_encoder.fit_transform(y).astype(int)
        X = X[:,-self.seqLen*100:-1:self.downSamplingFactor,:]
        # Now, we have the signals and the labels
        signals = X.reshape((nTotal,-1)) # nTotal x N (CS notation)
        labels = y
        rP = np.random.permutation(nTotal)
        # Split and save them
        self.samples = {}
        self.samples['train'] = {}
        self.samples['train']['signals'] = signals[rP[0:nTrain], :]
        self.samples['train']['labels'] = labels[rP[0:nTrain]]
        self.samples['valid'] = {} 
        self.samples['valid']['signals'] = signals[rP[nTrain:nTrain+nValid],:]
        self.samples['valid']['labels'] = labels[rP[nTrain:nTrain+nValid]]
        self.samples['test'] = {}
        self.samples['test']['signals'] = signals[rP[nTrain+nValid:nTotal],:]
        self.samples['test']['labels'] = labels[rP[nTrain+nValid:nTotal]]
        # Change data to specified type and device
        self.astype(self.dataType)
        self.to(self.device)

    def getSamples(self, samplesType, *args):
        # type: train, valid, test
        # args: 0 args, give back all
        # args: 1 arg: if int, give that number of samples, chosen at random
        # args: 1 arg: if list, give those samples precisely.
        # Check that the type is one of the possible ones
        assert samplesType == 'train' or samplesType == 'valid' \
                    or samplesType == 'test'
        # Check that the number of extra arguments fits
        assert len(args) <= 1
        # If there are no arguments, just return all the desired samples
        x = self.samples[samplesType]['signals']
        y = self.samples[samplesType]['labels']
        # If there's an argument, we have to check whether it is an int or a
        # list
        if len(args) == 1:
            # If it is an int, just return that number of randomly chosen
            # samples.
            if type(args[0]) == int:
                nSamples = x.shape[0] # total number of samples
                # We can't return more samples than there are available
                assert args[0] <= nSamples
                # Randomly choose args[0] indices
                selectedIndices = np.random.choice(nSamples, size = args[0],
                                                   replace = False)
                # The reshape is to avoid squeezing if only one sample is
                # requested
                x = x[selectedIndices,:].reshape([args[0], x.shape[1]])
                y = y[selectedIndices]
            else:
                # The fact that we put else here instead of elif type()==list
                # allows for np.array to be used as indices as well. In general,
                # any variable with the ability to index.
                x = x[args[0], :]
                # If only one element is selected, avoid squeezing. Given that
                # the element can be a list (which has property len) or an
                # np.array (which doesn't have len, but shape), then we can
                # only avoid squeezing if we check that it has been sequeezed
                # (or not)
                if len(x.shape) == 1:
                    x = x.reshape([1, x.shape[0]])
                # And assign the labels
                y = y[args[0]]

        return x, y

    def astype(self, dataType):
        if repr(dataType).find('torch') == -1:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                            = dataType(self.samples[key][secondKey])
        else:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                         = torch.tensor(self.samples[key][secondKey]).type(dataType)

        if dataType is not self.dataType:
            self.dataType = dataType

    def to(self, device):
        # This can only be done if they are torch tensors
        if repr(self.dataType).find('torch') >= 0:
            for key in self.samples.keys():
                for secondKey in self.samples[key].keys():
                    self.samples[key][secondKey] \
                                       = self.samples[key][secondKey].to(device)

            # If the device changed, save it.
            if device is not self.device:
                self.device = device

    def evaluate(self, yHat, y, tol = 1e-9):
        """
        Return the accuracy (ratio of yHat = y)
        """
        N = len(y)
        if 'torch' in repr(self.dataType):
            #   We compute the target label (hardmax)
            yHat = torch.argmax(yHat, dim = 1).type(self.dataType)
#            print(yHat)
#            print(y.type(torch.float64))
            #   And compute the error
            totalErrors = torch.sum(torch.abs(yHat - y.type(torch.float64)) > tol)
            accuracy = 1 - totalErrors.type(self.dataType)/N
        else:
            yHat = np.array(yHat)
            y = np.array(y)
            #   We compute the target label (hardmax)
            yHat = np.argmax(yHat, axis = 1).astype(y.dtype)
            #   And compute the error
            totalErrors = np.sum(np.abs(yHat - y) > tol)
            accuracy = 1 - totalErrors.astype(self.dataType)/N
        #   And from that, compute the accuracy
        return accuracy