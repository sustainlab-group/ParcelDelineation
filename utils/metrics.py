import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from keras import backend as K
#from segmentation_models.metrics import iou_score, dice_score

def get_metrics(y_true, y_pred, binarized=True):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    assert(y_true.shape == y_pred.shape)
    if not binarized:
      y_pred[y_pred > 0.5] = 1
      y_pred[y_pred != 1] = 0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
    acc = accuracy_score(y_true, y_pred)
    print('Dice/ F1 score:', f1)
    print('Accuracy score:', acc)
    print("Precision recall fscore", precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=1))
    return f1, acc

# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def f1(y_true, y_pred):
    import keras.backend as k
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#Simulating Dice coefficient using keras backend
def dice_coef_sim(y_true, y_pred, smooth=1):
    import keras.backend as k
    y_pred_bool = K.greater(y_pred, 0.5)                                                         
    y_pred = K.cast(y_pred_bool, K.floatx())                                                     
    intersection = K.sum(K.abs(y_true * y_pred))                                                 
    return (intersection / (K.sum(y_true) + K.sum(y_pred) - intersection + 1e-4)) 


#https://jaidevd.github.io/posts/weighted-loss-functions-for-instance-segmentation/
def iou(masks_true, masks_pred):
    """
    Get the IOU between each predicted mask and each true mask.

    Parameters
    ----------

    masks_true : array-like
        A 3D array of shape (n_true_masks, image_height, image_width)
    masks_pred : array-like
        A 3D array of shape (n_predicted_masks, image_height, image_width)

    Returns
    -------
    array-like
        A 2D array of shape (n_true_masks, n_predicted_masks), where
        the element at position (i, j) denotes the IoU between the `i`th true
        mask and the `j`th predicted mask.

    """
    if masks_true.shape[1:] != masks_pred.shape[1:]:
        raise ValueError('Predicted masks have wrong shape!')
    n_true_masks, height, width = masks_true.shape
    n_pred_masks = masks_pred.shape[0]
    m_true = masks_true.copy().reshape(n_true_masks, height * width).T
    m_pred = masks_pred.copy().reshape(n_pred_masks, height * width)
    numerator = np.dot(m_pred, m_true)
    denominator = m_pred.sum(1).reshape(-1, 1) + m_true.sum(0).reshape(1, -1)

    return numerator / (denominator - numerator)

#https://jaidevd.github.io/posts/weighted-loss-functions-for-instance-segmentation/
def evaluate_image_iou_metrics(masks_true, masks_pred, thresholds=0.5):
    """
    Get the average precision for the true and predicted masks of a single image,
    averaged over a set of thresholds

    Parameters
    ----------
    masks_true : array-like
        A 3D array of shape (n_true_masks, image_height, image_width)
    masks_pred : array-like
        A 3D array of shape (n_predicted_masks, image_height, image_width)

    Returns
    -------
    float
        The mean average precision of intersection over union between
        all pairs of true and predicted region masks.

    """
    int_o_un = iou(masks_true, masks_pred)
    benched = int_o_un > thresholds
    tp = benched.sum(-1).sum(-1)  # noqa
    fp = (benched.sum(2) == 0).sum(1)
    fn = (benched.sum(1) == 0).sum(1)

    return np.mean(tp / (tp + fp + fn))

def cosine_similarity(predictions, doc_embeddings):
    import keras.backend as k
    
    num = K.sum(predictions * doc_embeddings, axis=-1,keepdims=True)
    den = K.sqrt(K.sum(predictions * predictions, axis=-1, keepdims=True)) * K.sqrt(K.sum(doc_embeddings * doc_embeddings, axis=-1, keepdims=True))
    loss = -K.mean(num / den, axis=-1, keepdims=True)
    return loss

def L1_distance(merged_embeddings, dummy_vector):
    loss = K.mean(K.sum(K.abs(merged_embeddings - dummy_vector), axis=-1, keepdims=True))

    return loss
