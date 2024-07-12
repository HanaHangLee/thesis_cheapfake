import numpy as np
from sklearn.metrics import roc_auc_score





def calculate_metric(gtnp, pdnp):
    # input are numpy vector
    o_pdnp = np.copy(pdnp) # this is for AUROC score
    pdnp[pdnp>=0.5] = 1
    pdnp[pdnp!=1] = 0
    total_samples = len(gtnp)
    #print(f"Total sample: {total_samples}")
    total_correct = np.sum(gtnp == pdnp)
    accuracy = total_correct / total_samples
    gt_pos = np.where(gtnp == 1)[0]
    gt_neg = np.where(gtnp == 0)[0]
    TP = np.sum(pdnp[gt_pos])
    TN = np.sum(1 - pdnp[gt_neg])
    FP = np.sum(pdnp[gt_neg])
    FN = np.sum(1 - pdnp[gt_pos])
    precision = TP / (TP+FP)
    recall = TP/(TP+FN)
    f1 = 2*precision*recall/(precision+recall)
    metrics = {}
    metrics['accuracy'] = accuracy
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    metrics['tp'] = int(TP)
    metrics['tn'] = int(TN)
    metrics['fp'] = int(FP)
    metrics['fn'] = int(FN)
    try:
        metrics['auc'] = roc_auc_score(gtnp, o_pdnp)
    except Exception as e:
        print(e)
        metrics['auc'] = -1
    return metrics