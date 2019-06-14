import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    predictions -= np.max(predictions)
    e = np.exp(predictions)
    probs = e/np.sum(e)
    
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    loss = -(np.log(probs[target_index]))
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    
    # Make an array from target_index if it is not array
    if not isinstance(target_index, np.ndarray):
        target_index = np.array([[target_index]])
    target_index.shape = (-1,)
    
    # Determine batch size and number of features
    N = predictions.shape[-1]
    batch_size = target_index.shape[0]
    
    # Prepare predictions and make them of proper shape
    pred_copy = np.copy(predictions).reshape(batch_size, N)
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
    dprediction = np.copy(probs)
    dprediction[s, target_index] -= 1
    dprediction /= batch_size # normalize gradient
    
    #print("dprediction:")
    #print(dprediction)
    
    # Make gradiens shape match the one of prediction
    dprediction.shape = predictions.shape
    
    #raise Exception()
    
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    loss = reg_strength*np.tensordot(W,W)
    grad = 2*reg_strength*W
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
        
    predictions = np.dot(X, W)
    
    # TODO implement prediction and gradient over W
    #Determine batch size, number of features and number of classes
    batch_size = X.shape[0]
    N_f = X.shape[1]
    N_c = W.shape[1]
    
    #print('predistions:', predictions.shape, '\n')
    
    #Compute loss and gradient
    loss, grad = softmax_with_cross_entropy(predictions, target_index)
    
    #Compute loss and gradient of weight by loss
    dW = np.sum(grad.reshape(batch_size, 1, N_c)*X.reshape(batch_size, N_f, 1), axis=0)
    dW.shape = W.shape
    
    #print('dW:', dW, '\n\n')
    
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            raise Exception("Not implemented!")

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
