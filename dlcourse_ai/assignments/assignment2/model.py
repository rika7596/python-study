import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.layer_1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu = ReLULayer()
        self.layer_2 = FullyConnectedLayer(hidden_layer_size, n_output)
        
        self.hidden_layer_size = hidden_layer_size
        self.n_input = n_input
        self.n_output = n_output
        
        #raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        for v in self.params().values():
            v.grad.fill(0)
        
        # Hint: using self.params() might be useful!
        #raise Exception("Not implemented!")
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        #Compute FullyConnectedLayer_1 
        l_1 = self.layer_1.forward(X)
        #Compute ReLuLayer 
        l_relu = self.relu.forward(l_1)
        #Compute FullyConnectedLayer_2
        l_2 = self.layer_2.forward(l_relu)
        
        #compute loss and grad of F
        loss, grad_pred = softmax_with_cross_entropy(l_2, y)
        
        for v in self.params().values():
            l2_loss, l2_grad = l2_regularization(v.value, self.reg)
            loss += l2_loss
            v.grad += l2_grad
        
        grad_l_2 = self.layer_2.backward(grad_pred)
        grad_relu = self.relu.backward(grad_l_2)
        grad_l_1 = self.layer_1.backward(grad_relu)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        #raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        
        #Compute FullyConnectedLayer_1 
        l_1 = self.layer_1.forward(X)
        #Compute ReLuLayer 
        l_relu = self.relu.forward(l_1)
        #Compute FullyConnectedLayer_2
        l_2 = self.layer_2.forward(l_relu)
        
        #Compute pred
        pred = np.argmax(l_2, axis=1)
        #raise Exception("Not implemented!")
        return pred

    def params(self):
        p1 = self.layer_1.params()
        p2 = self.layer_2.params()
        result = {"W1": p1["W"], "B1": p1["B"], "W2": p2["W"], "B2": p2["B"]}
        #result = {'W1': self.w1, 'W2': self.w2, 'B1': self.b1, 'B2': self.b2}

        # TODO Implement aggregating all of the params
        
        #raise Exception("Not implemented!")

        return result
