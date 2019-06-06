import numpy as np

def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    TP = np.sum((prediction.astype(np.uint8) + ground_truth.astype(np.uint8))//2)
    FP = np.sum((prediction > ground_truth).astype(np.uint8))
    FN = np.sum((prediction < ground_truth).astype(np.uint8))
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    accuracy = np.sum((prediction == ground_truth).astype(np.uint8))/len(ground_truth)
    f1 = 2*precision*recall/(precision+recall)
    
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels
    
    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    accuracy = np.sum(prediction == ground_truth)/len(ground_truth)
    # TODO: Implement computing accuracy
    return accuracy
