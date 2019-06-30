import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength*np.linalg.norm(W)
    grad = 2*reg_strength*W
    #raise Exception("Not implemented!")
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    
    # Make an array from target_index if it is not array
    if not isinstance(target_index, np.ndarray):
        target_index = np.array([[target_index]])
    target_index.shape = (-1,)
    
    # Determine batch size and number of features
    N = preds.shape[-1]
    batch_size = target_index.shape[0]
    
    # Prepare predictions and make them of proper shape
    pred_copy = np.copy(preds).reshape(batch_size, N)
    pred_copy -= np.amax(pred_copy, axis=1).reshape(batch_size, 1)
    
    # Compute softmax
    e = np.exp(pred_copy)
    probs = e/np.sum(e, axis=1).reshape(batch_size, 1)
    
    # Compute loss
    s = np.arange(batch_size) # make array of [0, 1, 2, ..., (batch_size-1)]
    target_probs = probs[s, target_index] # using multiple array indexing
    
    loss = -np.average(np.log(target_probs))
    
    #print("loss:", loss)
    
    # Compute gradient
    d_preds = np.copy(probs)
    d_preds[s, target_index] -= 1
    d_preds /= batch_size # normalize gradient
    
    #print("dprediction:")
    #print(dprediction)
    
    # Make gradiens shape match the one of prediction
    d_preds.shape = preds.shape
    
    #raise Exception()
    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        relu = np.clip(X, a_min=0, a_max=None)
        self.X = X
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        #raise Exception("Not implemented!")
        return relu

    def backward(self, d_out):
        """
        Backward pass
        
        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        d_result = np.multiply((self.X>0),d_out)
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        self.X = X
        y_pred = np.dot(self.X, self.W.value) + self.B.value
        #y_pred = np.tensordot(self.X, self.W, axes=((1,),(0,)))+self.B
        # Your final implementation shouldn't have any loops
        #raise Exception("Not implemented!")
        return y_pred

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute
        d_input = np.dot(d_out, np.transpose(self.W.value))
        self.W.grad += np.tensordot(self.X, d_out, axes=(0,0))
        self.B.grad += np.sum(d_out, axis=0)
        # It should be pretty similar to linear classifier from
        # the previous assignment
        #raise Exception("Not implemented!")

        return d_input

    def params(self):
        return {'W': self.W, 'B': self.B}
