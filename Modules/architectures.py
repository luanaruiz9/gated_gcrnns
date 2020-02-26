
import numpy as np
import torch
import torch.nn as nn

import Utils.graphML as gml

zeroTolerance = 1e-9 # Values below this number are considered zero.

class SelectionGNN(nn.Module):
    """
    SelectionGNN: implement the selection GNN architecture

    Initialization:

        SelectionGNN(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
                     nonlinearity, # Nonlinearity
                     nSelectedNodes, poolingFunction, poolingSize, # Pooling
                     dimLayersMLP, # MLP in the end
                     GSO) # Structure

        Input:
            dimNodeSignals (list of int): dimension of the signals at each layer
            nFilterTaps (list of int): number of filter taps on each layer
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nFilterTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nFilterTaps) = L.
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            GSO (np.array): graph shift operator of choice.

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        SelectionGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nFilterTaps) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nFilterTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nFilterTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nFilterTaps # Filter taps
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        # Store the rest of the variables
        self.S = torch.tensor(GSO)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        gfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            gfl.append(gml.GraphFilter(self.F[l], self.F[l+1], self.K[l],
                                              self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            gfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            gfl.append(self.sigma())
            #\\ Pooling
            gfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            gfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.GFL = nn.Sequential(*gfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.GFL(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.GFL[3*l].addGSO(self.S)
            self.GFL[3*l+2].addGSO(self.S)

class SpectralGNN(nn.Module):
    """
    SpectralGNN: implement the selection GNN architecture using spectral filters

    Initialization:

        SpectralGNN(dimNodeSignals, nCoeff, bias, # Graph Filtering
                    nonlinearity, # Nonlinearity
                    nSelectedNodes, poolingFunction, poolingSize, # Pooling
                    dimLayersMLP, # MLP in the end
                    GSO) # Structure

        Input:
            dimNodeSignals (list of int): dimension of the signals at each layer
            nCoeff (list of int): number of coefficients on each layer
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nCoeff[l] is the number of coefficients for the
                filters implemented at layer l+1, thus len(nCoeff) = L.
            >> Obs.: If nCoeff[l] is less than the size of the graph, the
                remaining coefficients are interpolated by means of a cubic
                spline.
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            GSO (np.array): graph shift operator of choice.

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        SpectralGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nCoeff, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nCoeff) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nCoeff)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nCoeff)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nCoeff) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.M = nCoeff # Filter taps
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        self.S = torch.tensor(GSO)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        sgfl = [] # Graph Filtering Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            sgfl.append(gml.SpectralGF(self.F[l], self.F[l+1], self.M[l],
                                              self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            sgfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            sgfl.append(self.sigma())
            #\\ Pooling
            sgfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            sgfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.SGFL = nn.Sequential(*sgfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.SGFL(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.SGFL[3*l].addGSO(self.S)
            self.SGFL[3*l+2].addGSO(self.S)

class NodeVariantGNN(nn.Module):
    """
    NodeVariantGNN: implement the selection GNN architecture using node variant
        graph filters

    Initialization:

        NodeVariantGNN(dimNodeSignals, nShiftTaps, nNodeTaps, bias, # Filtering
                       nonlinearity, # Nonlinearity
                       nSelectedNodes, poolingFunction, poolingSize, # Pooling
                       dimLayersMLP, # MLP in the end
                       GSO) # Structure

        Input:
            dimNodeSignals (list of int): dimension of the signals at each layer
            nShiftTaps (list of int): number of shift taps on each layer
            nNodeTaps (list of int): number of node taps on each layer
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nShiftTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nShiftTaps) = L.
            >> Obs.: The length of the nShiftTaps and nNodeTaps has to be the
                same, and every element of one list is associated with the
                corresponding one on the other list to create the appropriate
                NVGF filter at each layer.
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            GSO (np.array): graph shift operator of choice.

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        NodeVariantGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nShiftTaps, nNodeTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than the number of
        # filter taps (because of the input number of features)
        assert len(dimNodeSignals) == len(nShiftTaps) + 1
        # The length of the shift taps list should be equal to the length of the
        # node taps list
        assert len(nShiftTaps) == len(nNodeTaps)
        # nSelectedNodes should be a list of size nShiftTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nShiftTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nShiftTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nShiftTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nShiftTaps # Filter Shift taps
        self.M = nNodeTaps # Filter node taps
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        self.S = torch.tensor(GSO)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        nvgfl = [] # Node Variant GF Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            nvgfl.append(gml.NodeVariantGF(self.F[l], self.F[l+1],
                                           self.K[l], self.M[l],
                                           self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            nvgfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            nvgfl.append(self.sigma())
            #\\ Pooling
            nvgfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            nvgfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.NVGFL = nn.Sequential(*nvgfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.NVGFL(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.NVGFL[3*l].addGSO(self.S)
            self.NVGFL[3*l+2].addGSO(self.S)

class EdgeVariantGNN(nn.Module):
    """
    EdgeVariantGNN: implement the selection GNN architecture using edge variant
        graph filters (through masking, not placement)

    Initialization:

        EdgeVariantGNN(dimNodeSignals, nShiftTaps, nFilterNodes, bias,
                       nonlinearity, # Nonlinearity
                       nSelectedNodes, poolingFunction, poolingSize,
                       dimLayersMLP, # MLP in the end
                       GSO) # Structure

        Input:
            dimNodeSignals (list of int): dimension of the signals at each layer
            nShiftTaps (list of int): number of shift taps on each layer
            nFilterNodes (list of int): number of nodes selected for the EV part
                of the hybrid EV filtering (recall that the first ones in the
                given permutation of S are the nodes selected)
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nShiftTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nShiftTaps) = L.
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            GSO (np.array): graph shift operator of choice.

        Output:
            nn.Module with a Selection GNN architecture with the above specified
            characteristics.

    Forward call:

        EdgeVariantGNN(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph filtering
                 dimNodeSignals, nShiftTaps, nFilterNodes, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than the number of
        # filter taps (because of the input number of features)
        assert len(dimNodeSignals) == len(nShiftTaps) + 1
        # Filter nodes is a list of int with the number of nodes to select for
        # the EV part at each layer; it should have the same length as the
        # number of filter taps
        assert len(nFilterNodes) == len(nShiftTaps)
        # nSelectedNodes should be a list of size nShiftTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nShiftTaps)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nShiftTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nShiftTaps) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nShiftTaps # Filter Shift taps
        self.M = nFilterNodes
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.bias = bias # Boolean
        self.S = torch.tensor(GSO)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        evgfl = [] # Node Variant GF Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            evgfl.append(gml.EdgeVariantGF(self.F[l], self.F[l+1],
                                            self.K[l], self.M[l], self.N[0],
                                            self.E, self.bias))
            # There is a 3*l below here, because we have three elements per
            # layer: graph filter, nonlinearity and pooling, so after each layer
            # we're actually adding elements to the (sequential) list.
            evgfl[3*l].addGSO(self.S)
            #\\ Nonlinearity
            evgfl.append(self.sigma())
            #\\ Pooling
            evgfl.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            # Same as before, this is 3*l+2
            evgfl[3*l+2].addGSO(self.S)
        # And now feed them into the sequential
        self.EVGFL = nn.Sequential(*evgfl) # Graph Filtering Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph filtering layer
        y = self.EVGFL(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.EVGFL[3*l].addGSO(self.S)
            self.EVGFL[3*l+2].addGSO(self.S)

class AggregationGNN(nn.Module):
    """
    AggregationGNN: implement the aggregation GNN architecture

    Initialization:

        Input:
            dimFeatures (list of int): number of features on each layer
            nFilterTaps (list of int): number of filter taps on each layer
            bias (bool): include bias after graph filter on every layer
            >> Obs.: dimFeatures[0] is the number of features (the dimension
                of the node signals) of the data, where dimFeatures[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimFeatures) = L+1. Slightly
                different, nFilterTaps[l] is the number of filter taps for the
                filters implemented at layer l+1, thus len(nFilterTaps) = L.
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            poolingFunction (torch.nn): module from torch.nn pooling layers
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            GSO (np.array): graph shift operator of choice.
            maxN (int): maximum number of neighborhood exchanges (default: None)
            >> Obs.: The node selected to carry out the aggregation is that one
                corresponding to the first element in the provided GSO.

        Output:
            nn.Module with an Aggregation GNN architecture with the above
            specified characteristics.

    Forward call:

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """
    def __init__(self,
                 # Graph filtering
                 dimFeatures, nFilterTaps, bias,
                 # Nonlinearity
                 nonlinearity,
                 # Pooling
                 poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO, maxN = None):
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimFeatures) == len(nFilterTaps) + 1
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nFilterTaps)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nFilterTaps) # Number of graph filtering layers
        self.F = dimFeatures # Features
        self.K = nFilterTaps # Filter taps
        self.E = GSO.shape[0]
        self.bias = bias # Boolean
        self.S = torch.tensor(GSO)
        self.sigma = nonlinearity
        self.rho = poolingFunction
        self.alpha = poolingSize # This acts as both the kernel_size and the
        # stride, so there is no overlap on the elements over which we take
        # the maximum (this is how it works as default)
        self.dimLayersMLP = dimLayersMLP
        # Maybe we don't want to aggregate information all the way to the end,
        # but up to some pre-specificed value maxN (for numerical reasons,
        # mostly)
        if maxN is None:
            self.maxN = GSO.shape[1]
        else:
            self.maxN = maxN if maxN < GSO.shape[1] else GSO.shape[1]
        # Let's also record the number of nodes on each layer (L+1, actually)
        self.N = [self.maxN]
        for l in range(self.L):
            # In pyTorch, the convolution is a valid correlation, instead of a
            # full one, which means that the output is smaller than the input.
            # Precisely, this smaller (check documentation for nn.conv1d)
            outConvN = self.N[l] - (self.K[l] - 1) # Size of the conv output
            # The next equation to compute the number of nodes is obtained from
            # the maxPool1d help in the pytorch documentation
            self.N += [int(
                            (outConvN - (self.alpha[l]-1) - 1)/self.alpha[l] + 1
                                    )]
            # int() on a float always applies floor()
        # Now, compute the necessary matrix. Recall that we want to build the
        # vector [[x]_{0}, [Sx]_{0}, [S^2x]_{0}, ..., [S^{N-1}x]_{0}]. But
        # instead of computing the powers of S^k and then keeping the 0th row,
        # we will multiply S with a delta = [1, 0, ..., 0]^T and keep each
        # result in the row.
        delta = np.zeros([self.E, GSO.shape[1], 1]) # E x N x 1
        delta[:, 0, 0] = 1. # E x N x 1
        # And create the place where to store all of this
        SN = delta.copy() # E x N x 1
        for k in range(1, self.maxN):
            delta = GSO @ delta
            SN = np.concatenate((SN, delta), axis = 2) # E x N x k
        # This matrix SN just needs to multiply the incoming x to obtain the
        # aggregated vector. And that's it.
        self.SN = torch.tensor(SN)
        # And now, we're finally ready to create the architecture:
        #\\\ Graph filtering layers \\\
        # OBS.: We could join this for with the one before, but we keep separate
        # for clarity of code.
        convl = [] # Convolutional Layers
        for l in range(self.L):
            #\\ Graph filtering stage:
            convl.append(nn.Conv1d(self.F[l], self.F[l+1], self.K[l],
                                   bias = self.bias))
            #\\ Nonlinearity
            convl.append(self.sigma())
            #\\ Pooling
            convl.append(self.rho(self.alpha[l]))
        # And now feed them into the sequential
        self.ConvLayers = nn.Sequential(*convl) # Convolutional layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.SN.shape[1]
        # Let's do the aggregation step
        z = torch.matmul(x, self.SN)
        # Let's call the convolutional layers
        y = self.ConvLayers(z)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move to device the GSO and its related variables.
        self.S = self.S.to(device)
        self.SN = self.SN.to(device)
        
class MultiNodeAggregationGNN(nn.Module):
    """
    MultiNodeAggregationGNN: implement the multi-node aggregation GNN
        architecture

    Initialization:

        Input:
            nSelectedNodes (list of int): number of selected nodes on each
                outer layer
            nShifts (list of int): number of shifts to be done by the selected
                nodes on each outer layer
            dimFeatures (list of list of int): the external list corresponds to
                the outer layers, the inner list to how many features to process
                on each inner layer (the aggregation GNN on each node)
            nFilterTaps (list of list of int): the external list corresponds to
                the outer layers, the inner list to how many filter taps to
                consider on each inner layer (the aggregation GNN on each node)
            bias (bool): include bias after graph filter on every layer
            nonlinearity (torch.nn): module from torch.nn non-linear activations
            poolingFunction (torch.nn): module from torch.nn pooling layers
            poolingSize (list of list of int): the external list corresponds to
                the outer layers, the inner list to the size of the neighborhood
                to compute the summary from at each inner layer (the aggregation
                GNN on each node)
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after all the outer layers
                have been computes
            GSO (np.array): graph shift operator of choice.

        Output:
            nn.Module with a Multi-Node Aggregation GNN architecture with the
            above specified characteristics.

    Forward call:

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the 
                multi-node aggregation GNN; shape:
                batchSize x dimLayersMLP[-1]
    
    Example:
        
    We want to create a Multi-Node GNN with two outer layers (i.e. two rounds of
    exchanging information on the graph). In the first round, we select 10 
    nodes, and in the following round, we select 5. Then, we need to determine
    how many shifts (how further away) we are going to move information around.
    In the first round (first outer layer) we shift around 4 times, and in the
    second round, we shift around 8 times (i.e. we get info from up to the 
    4-hop neighborhood in the first round, and 8-hop neighborhood in the
    secound round.)

    nSelectedNodes = [10, 5]
    nShifts = [4, 8]
    
    At this point, we have finished determining the outer structure (the one
    that involves exchanging information with neighbors). Now, we need to
    determine how to process the data within each node (the aggregation GNN
    that happens at each node). Since we have two outer layers, each of these
    parameters will be a list containing two lists. Each of these two lists
    determines the parameters to use to process internally the data. All nodes
    will use the same structure during each round.
    
    Say that we step inside a single node. We start with the signal received
    at the first outer layer (r=0), i.e., we have a signal of length 
    nShifts[0] = 4. We want to process this signal with a two-layer CNN creating
    3 and 6 features, respectively, using 2 filter taps, and with a ReLU
    nonlinearity in between and a max-pooling of size 2. This will just give
    an output with 6 features. This processing occurs at all 
    nSelectedNodes[0] = 10 nodes. After the second round, we get a new signal,
    with 6 features, but of length nShifts[1] = 8 at each of the
    nSelectedNodes[1] = 5 nodes. Now we want to process it through a two-layer
    CNN with that creates 12 and 18 features, with filters of size 2, with
    ReLU nonlinearities (same as before) and a max pooling (same as before) of 
    size 2. The setting is
    
    dimFeatures = [[1, 3, 6], [6, 12, 18]]
    nFilterTaps = [[2, 2], [2, 2]]
    nonlinearity = nn.ReLU
    poolingFunction = nn.MaxPool1d
    poolingSize = [[2, 2], [2, 2]]
    
    Recall that between the last convolutional layer (internal) and the output
    to be shared across nodes, there is an MLP layer adapting the number of
    features to the expected number of features of the next layer.
    
    Once we have all dimFeatures[-1][-1] = 18 features, collected at all
    nSelectedNodes[-1] = 5, we collect this information in a vector and feed it
    through two fully-connected layers of size 20 and 10.
    
    dimLayersMLP = [20, 10]
    """
    def __init__(self,
                 # Outer Structure
                 nSelectedNodes, nShifts,
                 # Inner Structure
                 #  Graph filtering
                 dimFeatures, nFilterTaps, bias,
                 #  Nonlinearity
                 nonlinearity,
                 #  Pooling
                 poolingFunction, poolingSize,
                 #  MLP in the end
                 dimLayersMLP,
                 # Graph Structure
                 GSO):
        # Initialize parent class
        super().__init__()
        # Check that we have an adequate GSO
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        # And create a third dimension if necessary
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        self.N = GSO.shape[1] # Store the number of nodes
        # Now, the interesting thing is that dimFeatures, nFilterTaps, and
        # poolingSize are all now lists of lists, and all of them need to have
        # the same length.
        self.R = len(nSelectedNodes) # Number of outer layers
        self.P = nSelectedNodes # Number of nodes selected on each outer layer
        # Check that the number of selected nodes does not exceed the number
        # of total nodes.
        # TODO: Should we consider that the number of nodes might not be
        # nonincreasing?
        for r in range(self.R):
            if self.P[r] > self.N:
                # If so, just force it to be the number of nodes.
                self.P[r] = self.N
        assert len(nShifts) == self.R
        self.Q = nShifts # Number of shifts of each node on each outer layer
        assert len(dimFeatures) == len(nFilterTaps) == self.R
        assert len(poolingSize) == self.R
        self.F = dimFeatures # List of lists containing the number of features
            # at each inner layer of each outer layer
        # Note that we have to add how many features we want in the ``last''
        # AggGNN layer before going into the MLP layer. Here, I will just
        # mix in the number of last specified features, but there are a lot of
        # other options, like no MLP whatsoever at the end of each convolutional
        # layer. But, why not?
        # TODO: (This adds quite the number of parameters, it would be nice to
        # do some reasonable tests to check whether this MLPs are necessary or
        # not).
        self.F.append([dimFeatures[-1][-1]])
        self.K = nFilterTaps # List of lists containing the number of filter 
            # taps at each inner layer of each outer layer.
        self.bias = bias # Boolean to include bias or not
        self.sigma = nonlinearity # Pointwise nonlinear function to include on
            # each aggregation GNN
        self.rho = poolingFunction # To use on every aggregation GNN
        self.alpha = poolingSize # Pooling size on each aggregation GNN
        self.dimLayersMLP = dimLayersMLP # MLP for each inner aggregation GNN
        self.S = torch.tensor(GSO)
        # Now that there are several things to do next:
        # - The AggregationGNN module always selects the first node, so if we
        #   want to select the first R, then we have to reorder it ourselves
        #   before adding the GSO to each AggregationGNN structure
        # - A regular python list does not register the parameters of the 
        #   corresponding nn.Module leading to bugs and issues on optimization.
        #   For this the class nn.ModuleList() has been created. Unlike 
        #   nn.Sequential(), this class does not have a forward method, because
        #   they are not supposed to act in a cascade way, just to keep track of
        #   dynamically changing numbers of layers.
        # - Another interesting observation is that, preliminary experiments, 
        #   show that nn.ModuleList() is also capable of handling lists of 
        #   lists. And this is precisely what we need: the first element (the
        #   outer one) corresponds to each outer layer, and each one of these
        #   elements contains another list with the Aggregation GNNs
        #   corresponding to the number of selected nodes on each outer layer.
        
        #\\\ Ordering:
        # So, let us start with the ordering. P (the number of selected nodes)
        # determines how many different orders we need (it's just rotating
        # the indices so that each one of those P is first)
        # The order will be a list of lists, the outer list having as many 
        # elements as maximum of P.
        self.order = [list(range(self.N))] # This is the order for the first
        #   selected nodes which is, clearly, the identity order
        maxP = max(self.P) # Maximum number of nodes to consider
        for p in range(1, maxP):
            allNodes = list(range(self.N)) # Create a list of all the nodes in
            # order.
            allNodes.remove(p) # Get rid of the element that we need to put
            # first
            thisOrder = [p] #  Take the pth element, put it in a list
            thisOrder.extend(allNodes)
            # extend that list with all other nodes, except for the pth one.
            self.order.append(thisOrder) # Store this in the order list
        
        #\\\ Aggregation GNN stage:
        self.aggGNNmodules = nn.ModuleList() # List to hold the AggGNN modules
        # Create the inner modules
        for r in range(self.R):
            # Add the list of inner modules
            self.aggGNNmodules.append(nn.ModuleList())
            # And start going through the inner modules
            for p in range(self.P[r]):
                thisGSO = GSO[:,self.order[p],:][:,:,self.order[p]]
                # # Reorder the GSO so that the selected node comes first and 
                # is thus selected by the AggGNN module.
                # Create the AggGNN module:
                self.aggGNNmodules[r].append(
                        AggregationGNN(self.F[r], self.K[r], self.bias,
                                       self.sigma,
                                       self.rho, self.alpha[r],
                                       # Now, the number of features in the
                                       # output of this AggregationGNN has to
                                       # be equal to the number of input 
                                       # features required at the next AggGNN
                                       # layer.
                                       [self.F[r+1][0]],
                                       thisGSO, maxN = self.Q[r]))
        # And this should be it for the creation of the AggGNN layers of the
        # MultiNodeAggregationGNN architecture. We move onto one last MLP
        fc = []
        if len(self.dimLayersMLP) > 0:
            # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            dimInputMLP = self.P[-1] * self.F[-1][0]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigma())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.
        
    def forward(self, x):
        # Now we compute the forward call
        # Check all relative dimensions
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0][0]
        assert x.shape[2] == self.N
        
        # Create an empty vector to store the output of the AggGNN of each node
        y = torch.empty(0)
        # For each outer layer (except the last one, since in the last one we
        # do not have to zero-pad)
        for r in range(self.R-1):
            # For each node
            for p in range(self.P[r]):
                # Re-order the nodes so that the selected nodes goes first
                xReordered = x[:, :, self.order[p]]
                # Compute the output of each GNN
                thisOutput = self.aggGNNmodules[r][p](xReordered)
                # Add it to the corresponding nodes
                y = torch.cat((y,thisOutput.unsqueeze(2)), dim = 2)
            # After this, y is of size B x F x P[r], but if we need to keep 
            # going for other outer layers, we need to zero-pad so that we can
            # keep shifting around on the original graph
            if y.shape[2] < self.N:
                # We zero-pad
                zeroPad = torch.zeros(batchSize, y.shape[1], self.N-y.shape[2])
                zeroPad = zeroPad.type(y.dtype).to(y.device)
                # Save as x
                x = torch.cat((y, zeroPad), dim = 2)
                # and reset y
                y = torch.empty(0)
                # At this point, note that x (and, before, y) where in order: 
                # the first elements corresponds to the first one in the
                # original ordering and so on. This means that the self.order
                # stored for the MultiNode still holds
            else:
                # We selected all nodes, so we do not need to zero-pad
                x = y
                # Save as x, and reset y
                y = torch.empty(0)
        # Last layer: we do not need to zero pad afterwards, so we just compute
        # the output of the GNN for each node and store that
        for p in range(self.P[-1]):
            xReordered = x[:, :, self.order[p]]
            thisOutput = self.aggGNNmodules[-1][p](xReordered)
            y = torch.cat((y,thisOutput.unsqueeze(2)), dim = 2)
                
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1][-1] * self.P[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.
    
    def to(self, device):
        # First, we initialize as always.
        super().to(device)
        # And then, in particular, move each architecture (that it will
        # internally move the GSOs and neighbors and stuff)
        for r in range(self.R):
            for p in range(self.P[r]):
                self.aggGNNmodules[r][p].to(device)
            
class GraphAttentionNetwork(nn.Module):
    """
    GraphAttentionNetwork: implement the graph attention network architecture

    Initialization:

        GraphAttentionNetwork(dimNodeSignals, nAttentionHeads, # Graph Filtering
                              nonlinearity, # Nonlinearity
                              nSelectedNodes, poolingFunction, poolingSize,
                              dimLayersMLP, bias, # MLP in the end
                              GSO) # Structure

        Input:
            dimNodeSignals (list of int): dimension of the signals at each layer
            nAttentionHeads (list of int): number of attention heads on each
                layer
            >> Obs.: dimNodeSignals[0] is the number of features (the dimension
                of the node signals) of the data, where dimNodeSignals[l] is the
                dimension obtained at the output of layer l, l=1,...,L.
                Therefore, for L layers, len(dimNodeSignals) = L+1. Slightly
                different, nAttentionHeads[l] is the number of filter taps for
                the filters implemented at layer l+1, thus
                len(nAttentionHeads) = L.
            nonlinearity (torch.nn.functional): function from module
                torch.nn.functional for non-linear activations
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer
            >> Obs.: The selected nodes are the first nSelectedNodes[l] starting
                from the first element in the order specified by the given GSO
            poolingFunction (nn.Module in Utils.graphML): summarizing function
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the graph filters have
                been applied
            bias (bool): include bias after each MLP layer
            GSO (np.array): graph shift operator of choice.

        Output:
            nn.Module with a Graph Attention Network architecture with the
            above specified characteristics.

    Forward call:

        GraphAttentionNetwork(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data after being processed by the selection
                GNN; shape: batchSize x dimLayersMLP[-1]
    """

    def __init__(self,
                 # Graph attentional layer
                 dimNodeSignals, nAttentionHeads,
                 # Nonlinearity (nn.functional)
                 nonlinearity,
                 # Pooling
                 nSelectedNodes, poolingFunction, poolingSize,
                 # MLP in the end
                 dimLayersMLP, bias,
                 # Structure
                 GSO):
        # Initialize parent:
        super().__init__()
        # dimNodeSignals should be a list and of size 1 more than nFilter taps.
        assert len(dimNodeSignals) == len(nAttentionHeads) + 1
        # nSelectedNodes should be a list of size nFilterTaps, since the number
        # of nodes in the first layer is always the size of the graph
        assert len(nSelectedNodes) == len(nAttentionHeads)
        # poolingSize also has to be a list of the same size
        assert len(poolingSize) == len(nAttentionHeads)
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
        # Store the values (using the notation in the paper):
        self.L = len(nAttentionHeads) # Number of graph filtering layers
        self.F = dimNodeSignals # Features
        self.K = nAttentionHeads # Attention Heads
        self.E = GSO.shape[0] # Number of edge features
        self.N = [GSO.shape[1]] + nSelectedNodes # Number of nodes
        # See that we adding N_{0} = N as the number of nodes input the first
        # layer: this above is the list containing how many nodes are between
        # each layer.
        self.S = torch.tensor(GSO)
        self.sigma = nonlinearity # This has to be a nn.functional instead of
            # just a nn
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.dimLayersMLP = dimLayersMLP
        self.bias = bias
        # And now, we're finally ready to create the architecture:
        #\\\ Graph Attentional Layers \\\
        # OBS.: The last layer has to have concatenate False, whereas the rest
        # have concatenate True. So we go all the way except for the last layer
        gat = [] # Graph Attentional Layers
        if self.L > 1:
            # First layer (this goes separate because there are not attention
            # heads increasing the number of features)
            #\\ Graph attention stage:
            gat.append(gml.GraphAttentional(self.F[0], self.F[1], self.K[0],
                                            self.E, self.sigma, True))
            gat[0].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[0], self.N[1], self.alpha[0]))
            gat[1].addGSO(self.S)
            # All the next layers (attention heads appear):
            for l in range(1, self.L-1):
                #\\ Graph attention stage:
                gat.append(gml.GraphAttentional(self.F[l] * self.K[l-1],
                                                self.F[l+1], self.K[l],
                                                self.E, self.sigma, True))
                # There is a 2*l below here, because we have two elements per
                # layer: graph filter and pooling, so after each layer
                # we're actually adding elements to the (sequential) list.
                gat[2*l].addGSO(self.S)
                #\\ Pooling
                gat.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
                # Same as before, this is 2*l+1
                gat[2*l+1].addGSO(self.S)
            # And the last layer (set concatenate to False):
            #\\ Graph attention stage:
            gat.append(gml.GraphAttentional(self.F[self.L-1] * self.K[self.L-2],
                                            self.F[self.L], self.K[self.L-1],
                                            self.E, self.sigma, False))
            gat[2* (self.L - 1)].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[self.L-1], self.N[self.L],
                                self.alpha[self.L-1]))
            gat[2* (self.L - 1) +1].addGSO(self.S)
        else:
            # If there's only one layer, it just go straightforward, adding a
            # False to the concatenation and no increase in the input features
            # due to attention heads
            gat.append(gml.GraphAttentional(self.F[0], self.F[1], self.K[0],
                                            self.E, self.sigma, False))
            gat[0].addGSO(self.S)
            #\\ Pooling
            gat.append(self.rho(self.N[l], self.N[l+1], self.alpha[l]))
            gat[1].addGSO(self.S)
        # And now feed them into the sequential
        self.GAT = nn.Sequential(*gat) # Graph Attentional Layers
        #\\\ MLP (Fully Connected Layers) \\\
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            # The first layer has to connect whatever was left of the graph
            # signal, flattened.
            # NOTE: Because sigma is a functional, instead of the layer, then
            # we need to pick up the layer for the MLP part.
            if str(self.sigma).find('relu') >= 0:
                self.sigmaMLP = nn.ReLU()
            elif str(self.sigma).find('tanh') >= 0:
                self.sigmaMLP = nn.Tanh()
                
            dimInputMLP = self.N[-1] * self.F[-1]
            # (i.e., we have N[-1] nodes left, each one described by F[-1]
            # features which means this will be flattened into a vector of size
            # N[-1]*F[-1])
            fc.append(nn.Linear(dimInputMLP, dimLayersMLP[0], bias = self.bias))
            # The last linear layer cannot be followed by nonlinearity, because
            # usually, this nonlinearity depends on the loss function (for
            # instance, if we have a classification problem, this nonlinearity
            # is already handled by the cross entropy loss or we add a softmax.)
            for l in range(len(dimLayersMLP)-1):
                # Add the nonlinearity because there's another linear layer
                # coming
                fc.append(self.sigmaMLP())
                # And add the linear layer
                fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                    bias = self.bias))
        # And we're done
        self.MLP = nn.Sequential(*fc)
        # so we finally have the architecture.

    def forward(self, x):
        # Now we compute the forward call
        assert len(x.shape) == 3
        batchSize = x.shape[0]
        assert x.shape[1] == self.F[0]
        assert x.shape[2] == self.N[0]
        # Let's call the graph attentional layers
        y = self.GAT(x)
        # Flatten the output
        y = y.reshape(batchSize, self.F[-1] * self.N[-1])
        # And, feed it into the MLP
        return self.MLP(y)
        # If self.MLP is a sequential on an empty list it just does nothing.

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        # And all the other variables derived from it.
        for l in range(self.L):
            self.GAT[2*l].addGSO(self.S)
            self.GAT[2*l+1].addGSO(self.S)
            
class GatedGCRNNforRegression(nn.Module):
    """
    GatedGCRNNforRegression: implements the full GCRNN architecture, i.e.
    h_t = sigma(\hat{Q_t}(A(S)*x_t) + \check{Q_t}(B(S)*h_{t-1}))
    y_t = rho(C(S)*h_t)
    where:
     - h_t, x_t, y_t are the state, input and output variables respectively
     - sigma and rho are the state and output nonlinearities
     - \hat{Q_t} and \check{Q_t} are the input and forget gate operators, which could be time, node or edge gates (or
     time+node, time+edge)
     - A(S), B(S) and C(S) are the input-to-state, state-to-state and state-to-output graph filters
     In the regression version of the Gated GCRNN, y_t is a graph signal

    Initialization:

        GatedGCRNNforRegression(inFeatures, stateFeatures, inputFilterTaps,
             stateFilterTaps, stateNonlinearity,
             outputNonlinearity,
             dimLayersMLP,
             GSO,
             bias,
             time_gating=True,
             spatial_gating=None,
             mlpType = 'oneMlp'
             finalNonlinearity = None,			
             dimNodeSignals=None, nFilterTaps=None,
             nSelectedNodes=None, poolingFunction=None, poolingSize=None, maxN = None)

        Input:
            inFeatures (int): dimension of the input signal at each node
            stateFeatures (int): dimension of the hidden state at each node
            inputFilterTaps (int): number of input filter taps
            stateFilterTaps (int): number of state filter taps 
            stateNonlinearity (torch.nn): sigma, state nonlinearity in the GRNN cell
            outputNonlinearity (torch.nn): rho, module from torch.nn nonlinear activations
            dimLayersMLP (list of int): number of hidden units of the MLP layers
            GSO (np.array): graph shift operator
            bias (bool): include bias after graph filter on every layer
            time_gating (bool): time gating flag, default True
            spatial_gating (string): None, 'node' or 'edge'
            mlpType (string): either 'oneMlp' or 'multipMLP'; 'multipMLP' corresponds to local MLPs, one per node
            finalNonlinearity (torch.nn): nonlinearity to apply to y, if any (e.g. softmax for classification)
            dimNodeSignals (list of int): dimension of the signals at each layer of C(S) if it is a GNN
            nFilterTaps (list of int): number of filter taps on each layer of C(S) if it is a GNN
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer of the C(S) if it is a Selection GNN
            poolingFunction (nn.Module in Utils.graphML): summarizing function of C(S) if it is a GNN with pooling
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer if C(S) is a GNN with pooling
            maxN (int): maximum number of neighborhood exchanges if C(S) is an Aggregation GNN (default: None)

        Output:
            nn.Module with a full GRNN architecture, state + output neural networks,
            with the above specified characteristics.

    Forward call:

        GatedGCRNNforRegression(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x seqLength x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data of shape
                batchSize x seqLength x dimFeatures x numberNodes
    """

    def __init__(self,
                 # State GCRNN
                 inFeatures, stateFeatures, inputFilterTaps,
                 stateFilterTaps, stateNonlinearity,
                 # Output NN nonlinearity
                 outputNonlinearity,			
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO,
                 # Bias
                 bias,
                 # Gating
                 time_gating = True,
                 spatial_gating = None,
                 # Type of MLP, one for all nodes or one, local, per node
                 mlpType = 'oneMlp',
                 # Final nonlinearity, if any, to apply to y
                 finalNonlinearity = None,			
                 # Output NN filtering if output NN is GNN
                 dimNodeSignals=None, nFilterTaps=None,
                 # Output NN pooling is output NN is GNN with pooling
                 nSelectedNodes=None, poolingFunction=None, poolingSize=None, maxN = None):
        
        # Initialize parent:
        super().__init__()
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
						
        # Store the values for the state GRNN (using the notation in the paper):
        self.F_i = inFeatures # Input features
        self.K_i = inputFilterTaps # Filter taps of input filter
        self.F_h = stateFeatures # State features
        self.K_h = stateFilterTaps # Filter taps of state filter
        self.E = GSO.shape[0] # Number of edge features
        self.N = GSO.shape[1] # Number of nodes
        self.bias = bias # Boolean
        self.time_gating = time_gating # Boolean
        self.spatial_gating = spatial_gating # None, "node" or "edge"
        self.S = torch.tensor(GSO)
        self.sigma1 = stateNonlinearity	 
        # Declare State GCRNN				
        self.stateGCRNN = gml.GGCRNNCell(self.F_i, self.F_h, self.K_i,
                 self.K_h, self.sigma1, self.time_gating, self.spatial_gating,
                 self.E, self.bias)
        self.stateGCRNN.addGSO(self.S)
        # Dimensions of output GNN's  lfully connected layers or of the output MLP
        self.dimLayersMLP = dimLayersMLP
        # Output neural network nonlinearity
        self.sigma2 = outputNonlinearity
        # Selection/Aggregation GNN parameters for the output neural network (default None)
        self.F_o = dimNodeSignals
        self.K_o = nFilterTaps
        self.nSelectedNodes = nSelectedNodes
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.maxN = maxN
        # Nonlinearity to apply to the output, e.g. softmax for classification (default None)
        self.sigma3 = finalNonlinearity
        # Type of MLP
        self.mlpType = mlpType        
        
        #\\ If output neural network is MLP:
        if dimNodeSignals is None and nFilterTaps is None:
            fc = []
            if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
                if mlpType == 'oneMlp':
                    # The first layer has to connect whatever was left of the graph
                    # signal, flattened.
                    dimInputMLP = self.N * self.F_h
                    # (i.e., N nodes, each one described by F_h features,
                    # which means this will be flattened into a vector of size
                    # N x F_h)
                elif mlpType == 'multipMlp':
                    # one perceptron per node, same parameters across all of them
                    dimInputMLP = self.F_h
                fc.append(nn.Linear(dimInputMLP, self.dimLayersMLP[0], bias = self.bias))
                for l in range(len(dimLayersMLP)-1):
                    # Add the nonlinearity because there's another linear layer
                    # coming
                    fc.append(self.sigma2())
                    # And add the linear layer
                    fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                        bias = self.bias))   
            # And we're done
            # Declare output MLP
            if self.sigma3 is not None :   
                fc.append(self.sigma3())
            self.outputNN = nn.Sequential(*fc)
            # so we finally have the architecture.
            
        #\\ If the output neural network is an Aggregation GNN
        elif nSelectedNodes is None and poolingFunction is not gml.NoPool:
            agg = []
            self.F_o = dimNodeSignals
            self.K_o = nFilterTaps
            self.rho = poolingFunction
            self.alpha = poolingSize
            self.maxN = maxN
            # Declare output GNN
            outputNN = AggregationGNN(self.F_o, self.K_o, self.bias,
                                           self.sigma2, self.nSelectedNodes, self.rho, self.alpha, 
                                           self.dimLayersMLP, GSO)
            agg.append(outputNN)
            # Adding final nonlinearity, if any
            if self.sigma3 is not None:
                agg.append(self.sigma3())
            self.outputNN = nn.Sequential(*agg)
        
        #\\ If the output neural network is a Selection GNN
        else:
            sel = []
            self.F_o = dimNodeSignals
            self.K_o = nFilterTaps
            self.nSelectedNodes = nSelectedNodes
            self.rho = poolingFunction
            self.alpha = poolingSize
            # Declare Output GNN	
            outputNN = SelectionGNN(self.F_o, self.K_o, self.bias, 
                                         self.sigma2, self.nSelectedNodes, 
                                         self.rho, self.alpha, self.dimLayersMLP,
                                         GSO)
            sel.append(outputNN)
            # Adding final nonlinearity, if any
            if self.sigma3 is not None:
                sel.append(self.sigma3())
            self.outputNN = nn.Sequential(*sel)

    def forward(self, x, h0):
        # Now we compute the forward call
        batchSize = x.shape[0]
        seqLength = x.shape[1]
        H = self.stateGCRNN(x,h0)
        # flatten by merging batch and sequence length dimensions
        flatH = H.view(-1,self.F_h,self.N)

        if self.F_o is None: # outputNN is MLP
            if self.mlpType == 'multipMlp':
                flatH = flatH.view(-1,self.F_h,self.N)
                flatH = flatH.transpose(1,2) 
                flatY = torch.empty(0)
                for i in range (self.N):
                    hNode = flatH.narrow(1,i,1)
                    hNode = hNode.squeeze()
                    yNode = self.outputNN(hNode)
                    yNode = yNode.unsqueeze(1)
                    flatY = torch.cat([flatY,yNode],1)
                flatY = flatY.transpose(1,2)
                flatY = flatY.squeeze()
            elif self.mlpType == 'oneMlp':
                flatH = flatH.view(-1,self.F_h*self.N)
                flatY = self.outputNN(flatH)
        else:
            flatY = self.outputNN(flatH)
        # recover original batch and sequence length dimensions
        y = flatY.view(batchSize,seqLength,-1)
        y = torch.unsqueeze(y,2)
        return y

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)

class GatedGCRNNforClassification(nn.Module):
    """
    GatedGCRNNfforClassification: implements the full GCRNN architecture, i.e.
    h_t = sigma(\hat{Q_t}(A(S)*x_t) + \check{Q_t}(B(S)*h_{t-1}))
    y_t = rho(C(S)*h_t)
    where:
     - h_t, x_t, y_t are the state, input and output variables respectively
     - sigma and rho are the state and output nonlinearities
     - \hat{Q_t} and \check{Q_t} are the input and forget gate operators, which could be time, node or edge gates (or
     time+node, time+edge)
     - A(S), B(S) and C(S) are the input-to-state, state-to-state and state-to-output graph filters
     In the classification version of the Gated GCRNN, y_t is a C-dimensional one-hot vector, where C is the number of classes

    Initialization:

        GatedGCRNNforClassification(inFeatures, stateFeatures, inputFilterTaps,
             stateFilterTaps, stateNonlinearity,
             outputNonlinearity,
             dimLayersMLP,
             GSO,
             bias,
             time_gating=True,
             spatial_gating=None,
             mlpType = 'oneMlp'
             finalNonlinearity = None,			
             dimNodeSignals=None, nFilterTaps=None,
             nSelectedNodes=None, poolingFunction=None, poolingSize=None, maxN = None)

        Input:
            inFeatures (int): dimension of the input signal at each node
            stateFeatures (int): dimension of the hidden state at each node
            inputFilterTaps (int): number of input filter taps
            stateFilterTaps (int): number of state filter taps 
            stateNonlinearity (torch.nn): sigma, state nonlinearity in the GRNN cell
            outputNonlinearity (torch.nn): rho, module from torch.nn nonlinear activations
            dimLayersMLP (list of int): number of hidden units of the MLP layers
            GSO (np.array): graph shift operator
            bias (bool): include bias after graph filter on every layer
            time_gating (bool): time gating flag, default True
            spatial_gating (string): None, 'node' or 'edge'
            mlpType (string): either 'oneMlp' or 'multipMLP'; 'multipMLP' corresponds to local MLPs, one per node
            finalNonlinearity (torch.nn): nonlinearity to apply to y, if any (e.g. softmax for classification)
            dimNodeSignals (list of int): dimension of the signals at each layer of C(S) if it is a GNN
            nFilterTaps (list of int): number of filter taps on each layer of C(S) if it is a GNN
            nSelectedNodes (list of int): number of nodes to keep after pooling
                on each layer of the C(S) if it is a Selection GNN
            poolingFunction (nn.Module in Utils.graphML): summarizing function of C(S) if it is a GNN with pooling
            poolingSize (list of int): size of the neighborhood to compute the
                summary from at each layer if C(S) is a GNN with pooling
            maxN (int): maximum number of neighborhood exchanges if C(S) is an Aggregation GNN (default: None)

        Output:
            nn.Module with a full GRNN architecture, state + output neural networks,
            with the above specified characteristics.

    Forward call:

        GatedGCRNNforClassification(x)

        Input:
            x (torch.tensor): input data of shape
                batchSize x seqLength x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output data of shape
                batchSize x seqLength x dimFeatures x numberNodes
    """

    def __init__(self,
                 # State GCRNN
                 inFeatures, stateFeatures, inputFilterTaps,
                 stateFilterTaps, stateNonlinearity,
                 # Output NN nonlinearity
                 outputNonlinearity,			
                 # MLP in the end
                 dimLayersMLP,
                 # Structure
                 GSO,
                 # Bias
                 bias,
                 # Gating
                 time_gating = True,
                 spatial_gating = None,
                 # Final nonlinearity, if any, to apply to y
                 finalNonlinearity = None,			
                 # Output NN filtering if output NN is GNN
                 dimNodeSignals=None, nFilterTaps=None,
                 # Output NN pooling is output NN is GNN with pooling
                 nSelectedNodes=None, poolingFunction=None, poolingSize=None, maxN = None):
        
        # Initialize parent:
        super().__init__()
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
						
        # Store the values for the state GRNN (using the notation in the paper):
        self.F_i = inFeatures # Input features
        self.K_i = inputFilterTaps # Filter taps of input filter
        self.F_h = stateFeatures # State features
        self.K_h = stateFilterTaps # Filter taps of state filter
        self.E = GSO.shape[0] # Number of edge features
        self.N = GSO.shape[1] # Number of nodes
        self.bias = bias # Boolean
        self.time_gating = time_gating # Boolean
        self.spatial_gating = spatial_gating # None, "node" or "edge"
        self.S = torch.tensor(GSO)
        self.sigma1 = stateNonlinearity	 
        # Declare State GCRNN				
        self.stateGCRNN = gml.GGCRNNCell(self.F_i, self.F_h, self.K_i,
                 self.K_h, self.sigma1, self.time_gating, self.spatial_gating,
                 self.E, self.bias)
        self.stateGCRNN.addGSO(self.S)
        # Dimensions of output GNN's  lfully connected layers or of the output MLP
        self.dimLayersMLP = dimLayersMLP
        # Output neural network nonlinearity
        self.sigma2 = outputNonlinearity
        # Selection/Aggregation GNN parameters for the output neural network (default None)
        self.F_o = dimNodeSignals
        self.K_o = nFilterTaps
        self.nSelectedNodes = nSelectedNodes
        self.rho = poolingFunction
        self.alpha = poolingSize
        self.maxN = maxN
        # Nonlinearity to apply to the output, e.g. softmax for classification (default None)
        self.sigma3 = finalNonlinearity    
        
        #\\ If output neural network is MLP:
        if dimNodeSignals is None and nFilterTaps is None:
            fc = []
            if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
                # The first layer has to connect whatever was left of the graph
                # signal, flattened.
                dimInputMLP = self.N * self.F_h
                # (i.e., N nodes, each one described by F_h features,
                # which means this will be flattened into a vector of size
                    # N x F_h)
                fc.append(nn.Linear(dimInputMLP, self.dimLayersMLP[0], bias = self.bias))
                for l in range(len(dimLayersMLP)-1):
                    # Add the nonlinearity because there's another linear layer
                    # coming
                    fc.append(self.sigma2())
                    # And add the linear layer
                    fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                        bias = self.bias))   
            # And we're done
            # Declare output MLP
            if self.sigma3 is not None :   
                fc.append(self.sigma3())
            self.outputNN = nn.Sequential(*fc)
            # so we finally have the architecture.
            
        #\\ If the output neural network is an Aggregation GNN
        elif nSelectedNodes is None and poolingFunction is not gml.NoPool:
            agg = []
            self.F_o = dimNodeSignals
            self.K_o = nFilterTaps
            self.rho = poolingFunction
            self.alpha = poolingSize
            self.maxN = maxN
            # Declare output GNN
            outputNN = AggregationGNN(self.F_o, self.K_o, self.bias,
                                           self.sigma2, self.nSelectedNodes, self.rho, self.alpha, 
                                           self.dimLayersMLP, GSO)
            agg.append(outputNN)
            # Adding final nonlinearity, if any
            if self.sigma3 is not None:
                agg.append(self.sigma3())
            self.outputNN = nn.Sequential(*agg)
        
        #\\ If the output neural network is a Selection GNN
        else:
            sel = []
            self.F_o = dimNodeSignals
            self.K_o = nFilterTaps
            self.nSelectedNodes = nSelectedNodes
            self.rho = poolingFunction
            self.alpha = poolingSize
            # Declare Output GNN	
            outputNN = SelectionGNN(self.F_o, self.K_o, self.bias, 
                                         self.sigma2, self.nSelectedNodes, 
                                         self.rho, self.alpha, self.dimLayersMLP,
                                         GSO)
            sel.append(outputNN)
            # Adding final nonlinearity, if any
            if self.sigma3 is not None:
                sel.append(self.sigma3())
            self.outputNN = nn.Sequential(*sel)

    def forward(self, x, h0):
        # Now we compute the forward call
        H = self.stateGCRNN(x,h0)
        h = H.select(1,-1)
        if self.F_o is None: # outputNN is MLP
            h = h.view(-1,self.F_h*self.N)
            y = self.outputNN(h)
        else:
            y = self.outputNN(h)
        return y

    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S = self.S.to(device)
        
class RNNforRegression(nn.Module):
    """
    RNNforRegression: implements the full RNN architecture for graph signal regression

    Initialization:

        RNNforRegression(inFeatures, stateFeatures,
             stateNonlinearity,
             innerFeatures,
             kernelSizes,
             dimLayersMLP,
             outputNonlinearity,
             GSO,
             bias,
             finalNonlinearity = None)

        Input:
            inFeatures (int): dimension of the input signal at each node
            stateFeatures (int): dimension of the hidden state at each node
            stateNonlinearity (torch.nn): nonlinearity in the RNN cell
            innerFeatures(list of int) = number of features at each layer of CNN
            kernelSizes (list of in) = kernel sizes at each layer of CNN
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the CNN (valid for 
                both selections, needed for mlp)
            outputNonlinearity (torch.nn): module from torch.nn non-linear activations
            GSO (np.array): graph shift operator of choice.
            bias (bool): include bias after graph filter on every layer
            finalNonlinearity (torch.nn): nonlinearity to apply to the output, if any


        Output:
            nn.Module with a full RNN architecture, state + output networks,
            with the above specified characteristics.

    Forward call:

        RNNforRegression(x, h0)

        Input:
            x (torch.tensor): input data of shape
                batchSize x seqLength x dimFeatures x numberNodes
            h0 (torch.tensor): initial state of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output of the RNN computed from h and 
            transformed back to graph domain; 
            shape: batchSize x seqLength x dimFeatures x numberNodes
    """
    def __init__(self,
                 # State GRNN
                 inFeatures, stateFeatures,
                 stateNonlinearity,
                 # MLP in the end
                 dimLayersMLP,
                 # Output NN nonlinearity
                 outputNonlinearity,			
                 # Structure
                 GSO,
                 # Bias
                 bias,
                 # Final nonlinearity
                 finalNonlinearity = None):
        # Initialize parent:
        super().__init__()
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
    						
        # Store the values for the state RNN (using the notation in the paper):
        self.F_i = inFeatures # Input features
        self.F_h = stateFeatures # State features
        self.E = GSO.shape[0] # Number of edge features
        self.N = GSO.shape[1] # Number of nodes
        self.bias = bias # Boolean
        self.S_tensor = torch.tensor(GSO)
        self.S = GSO
        self.sigma1 = stateNonlinearity	 
        # Declare State RNN				
        self.RNN = torch.nn.RNN(self.N*self.F_i, self.F_h, num_layers=1,
                 nonlinearity=self.sigma1, bias=self.bias, batch_first=True)
        # Dimensions of output CNN's FC layers or of the output MLP
        self.dimLayersMLP = dimLayersMLP
        # Output NN nonlinearity
        self.sigma2 = outputNonlinearity
        # Nonlinearity to apply to the output (default None)
        self.sigma3 = finalNonlinearity      
        
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            dimInputMLP = self.F_h
            if len(self.dimLayersMLP) != 1:
                fc.append(nn.Linear(dimInputMLP, self.dimLayersMLP[0], bias = self.bias))
                for l in range(len(dimLayersMLP)-1):
                    # Add the nonlinearity because there's another linear layer
                    # coming
                    fc.append(self.sigma2())
                    # And add the linear layer
                    if l == len(dimLayersMLP)-2:
                        fc.append(nn.Linear(dimLayersMLP[-2], dimLayersMLP[-1]*self.N,
                                        bias = self.bias))
                    else:
                        fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                        bias = self.bias))  
            else:
                fc.append(nn.Linear(dimInputMLP, self.dimLayersMLP[0]*self.N, bias = self.bias))
        # And we're done
        if self.sigma3 is not None :   
            fc.append(self.sigma3())
        # Declare Output MLP
        self.outputNN = nn.Sequential(*fc)
        # so we finally have the architecture    
    

    def forward(self, x, h0, c0):
        # Now we compute the forward call
        batchSize = x.shape[0]
        seqLength = x.shape[1]
        x = x.view(batchSize,seqLength,-1)

        h0 = h0.view(batchSize,1,-1)
        h0 = h0.transpose(0,1)
        
        H,_ = self.RNN(x,h0)
        # flatten by merging batch and sequence length dimensions
        flatH = H.reshape(batchSize*seqLength,H.shape[2])
        #print(flatH.shape)
        flatY = self.outputNN(flatH)
        y = flatY.view(batchSize,seqLength,-1,self.N)
        #y = torch.matmul(y,v.transpose(0,1))
        return y
    
    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S_tensor = self.S_tensor.to(device)
        
class RNNforClassification(nn.Module):
    """
    RNNforClassification: implements the full RNN architecture for graph signal classification
    Initialization:

        RNNforClassification(inFeatures, stateFeatures,
             stateNonlinearity,
             innerFeatures,
             kernelSizes,
             dimLayersMLP,
             outputNonlinearity,
             GSO,
             bias,
             finalNonlinearity = None)

        Input:
            inFeatures (int): dimension of the input signal at each node
            stateFeatures (int): dimension of the hidden state at each node
            stateNonlinearity (torch.nn): nonlinearity in the RNN cell
            innerFeatures(list of int) = number of features at each layer of CNN
            kernelSizes (list of in) = kernel sizes at each layer of CNN
            dimLayersMLP (list of int): number of output hidden units of a
                sequence of fully connected layers after the CNN (valid for 
                both selections, needed for mlp)
            outputNonlinearity (torch.nn): module from torch.nn non-linear activations
            GSO (np.array): graph shift operator of choice.
            bias (bool): include bias after graph filter on every layer
            finalNonlinearity (torch.nn): nonlinearity to apply to the output, if any


        Output:
            nn.Module with a full RNN architecture, state + output networks,
            with the above specified characteristics.

    Forward call:

        RNNforClassification(x, h0)

        Input:
            x (torch.tensor): input data of shape
                batchSize x seqLength x dimFeatures x numberNodes
            h0 (torch.tensor): initial state of shape
                batchSize x dimFeatures x numberNodes

        Output:
            y (torch.tensor): output of the RNN computed from h and 
            transformed back to graph domain; 
            shape: batchSize x seqLength x dimFeatures x numberNodes
    """
    def __init__(self,
                 # State GRNN
                 inFeatures, stateFeatures,
                 stateNonlinearity,
                 # MLP in the end
                 dimLayersMLP,
                 # Output NN nonlinearity
                 outputNonlinearity,			
                 # Structure
                 GSO,
                 # Bias
                 bias,
                 # Final nonlinearity
                 finalNonlinearity = None):
        # Initialize parent:
        super().__init__()
        # Check whether the GSO has features or not. After that, always handle
        # it as a matrix of dimension E x N x N.
        assert len(GSO.shape) == 2 or len(GSO.shape) == 3
        if len(GSO.shape) == 2:
            assert GSO.shape[0] == GSO.shape[1]
            GSO = GSO.reshape([1, GSO.shape[0], GSO.shape[1]]) # 1 x N x N
        else:
            assert GSO.shape[1] == GSO.shape[2] # E x N x N
    						
        # Store the values for the state RNN (using the notation in the paper):
        self.F_i = inFeatures # Input features
        self.F_h = stateFeatures # State features
        self.E = GSO.shape[0] # Number of edge features
        self.N = GSO.shape[1] # Number of nodes
        self.bias = bias # Boolean
        self.S_tensor = torch.tensor(GSO)
        self.S = GSO
        self.sigma1 = stateNonlinearity	 
        # Declare State RNN				
        self.RNN = torch.nn.RNN(self.N*self.F_i, self.F_h, num_layers=1,
                 nonlinearity=self.sigma1, bias=self.bias, batch_first=True)
        # Dimensions of output CNN's FC layers or of the output MLP
        self.dimLayersMLP = dimLayersMLP
        # Output NN nonlinearity
        self.sigma2 = outputNonlinearity
        # Nonlinearity to apply to the output (default None)
        self.sigma3 = finalNonlinearity      
        
        fc = []
        if len(self.dimLayersMLP) > 0: # Maybe we don't want to MLP anything
            dimInputMLP = self.F_h
            if len(self.dimLayersMLP) != 1:
                fc.append(nn.Linear(dimInputMLP, self.dimLayersMLP[0], bias = self.bias))
                for l in range(len(dimLayersMLP)-1):
                    # Add the nonlinearity because there's another linear layer
                    # coming
                    fc.append(self.sigma2())
                    # And add the linear layer
                    if l == len(dimLayersMLP)-2:
                        fc.append(nn.Linear(dimLayersMLP[-2], dimLayersMLP[-1],
                                        bias = self.bias))
                    else:
                        fc.append(nn.Linear(dimLayersMLP[l], dimLayersMLP[l+1],
                                        bias = self.bias))  
            else:
                fc.append(nn.Linear(dimInputMLP, self.dimLayersMLP[0], bias = self.bias))
        # And we're done
        if self.sigma3 is not None :   
            fc.append(self.sigma3())
        # Declare Output MLP
        self.outputNN = nn.Sequential(*fc)
        # so we finally have the architecture    
    

    def forward(self, x, h0, c0):
        # Now we compute the forward call
        batchSize = x.shape[0]
        seqLength = x.shape[1]
        x = x.view(batchSize,seqLength,-1)

        h0 = h0.view(batchSize,1,-1)
        h0 = h0.transpose(0,1)
        
        H,_ = self.RNN(x,h0)
        h = H.select(1,-1)
        y = self.outputNN(h)
        #y = torch.matmul(y,v.transpose(0,1))
        return y
    
    def to(self, device):
        # Because only the filter taps and the weights are registered as
        # parameters, when we do a .to(device) operation it does not move the
        # GSOs. So we need to move them ourselves.
        # Call the parent .to() method (to move the registered parameters)
        super().to(device)
        # Move the GSO
        self.S_tensor = self.S_tensor.to(device)