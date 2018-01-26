from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import label_ranking_average_precision_score
from sklearn import metrics
from collections import Counter
import math
import numpy as np

def patk(predictions, labels):
    pak=np.zeros(3)
    K = np.array([1,3,5])
    for i in range(predictions.shape[0]):
        pos = np.argsort(-predictions[i,:])
        y=labels[i,:]
    y=y[pos]
    for j in range(3):
        k=K[j]
        pak[j]+=(np.sum(y[:k])/k)
    pak = pak/predictions.shape[0]
    return pak

def cm_precision_recall(prediction,truth):
    """Evaluate confusion matrix, precision and recall for given set of labels and predictions
     Args
       prediction: a vector with predictions
       truth: a vector with class labels
     Returns:
       cm: confusion matrix
       precision: precision score
       recall: recall score"""
    confusion_matrix = Counter()

    positives = [1]

    binary_truth = [x in positives for x in truth]
    binary_prediction = [x in positives for x in prediction]

    for t, p in zip(binary_truth, binary_prediction):
        confusion_matrix[t,p] += 1

    cm = np.array([confusion_matrix[True,True], confusion_matrix[False,False], confusion_matrix[False,True], confusion_matrix[True,False]])
    #print cm
    precision = (cm[0]/(cm[0]+cm[2]+0.000001))
    recall = (cm[0]/(cm[0]+cm[3]+0.000001))
    return cm, precision, recall

def bipartition_scores(labels,predictions):
  """ Computes bipartitation metrics for a given multilabel predictions and labels
      Args:
        logits: Logits tensor, float - [batch_size, NUM_LABELS].
        labels: Labels tensor, int32 - [batch_size, NUM_LABELS].
      Returns:
        bipartiation: an array with micro_precision, micro_recall, micro_f1,macro_precision, macro_recall, macro_f1"""
  sum_cm=np.zeros((4))
  macro_precision=0
  macro_recall=0
  for i in range(labels.shape[1]):
    truth=labels[:,i]
    prediction=predictions[:,i]
    cm,precision,recall=cm_precision_recall(prediction, truth)
    sum_cm+=cm
    macro_precision+=precision
    macro_recall+=recall
    
  macro_precision=macro_precision/labels.shape[1]
  macro_recall=macro_recall/labels.shape[1]
  macro_f1 = 2*(macro_precision)*(macro_recall)/(macro_precision+macro_recall+0.000001)
    
  micro_precision = sum_cm[0]/(sum_cm[0]+sum_cm[2]+0.000001)
  micro_recall=sum_cm[0]/(sum_cm[0]+sum_cm[3]+0.000001)
  micro_f1 = 2*(micro_precision)*(micro_recall)/(micro_precision+micro_recall+0.000001)
  bipartiation = np.asarray([micro_precision, micro_recall, micro_f1,macro_precision, macro_recall, macro_f1])
  return bipartiation

def evaluate(predictions, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_LABELS].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_LABELS).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  #labels = labels.astype(int)
  coverage= coverage_error(labels, predictions)
  average_precision = label_ranking_average_precision_score(labels, predictions)
  ranking_loss = label_ranking_loss(labels, predictions)
  pak = patk(predictions, labels)

  #predictions = predictions.astype(int)
  for i in range(predictions.shape[0]):
    predictions[i,:][predictions[i,:]>=0.5]=1
    predictions[i,:][predictions[i,:]<0.5]=0
 #print sum(predictions, axis=1)
  
  #micro_precision = metrics.precision_score(labels, predictions, average='micro')
  #micro_recall = metrics.recall_score(labels, predictions, average='micro')
  #micro_f1 = metrics.f1_score(labels, predictions, average='micro')

  #macro_precision = metrics.precision_score(labels, predictions, average='macro')
  #macro_recall = metrics.recall_score(labels, predictions, average='macro')
  #macro_f1 = metrics.f1_score(labels, predictions, average='macro')
  
  micro_precision, micro_recall, micro_f1,macro_precision, macro_recall, macro_f1 = bipartition_scores(labels, predictions)
  
  performance = np.asarray([coverage,average_precision,ranking_loss,micro_f1,macro_f1,micro_precision,micro_recall,macro_precision,macro_recall, pak[0], pak[1], pak[2]])
  return performance
