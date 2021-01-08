"""
evaluation
"""

import numpy as np

def evalROC(gold_scores, pred_scores):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(gold_scores, pred_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("roc_auc:", roc_auc)


def evalPR(gold_scores, pred_scores):
    from sklearn.metrics import precision_recall_curve, auc
    prec, recall, _ = precision_recall_curve(gold_scores, pred_scores, pos_label=1)
    pr_auc = auc(recall, prec)
    print("pr_auc:", pr_auc)


def tuneThreshold(gold_scores, pred_scores):
    from  sklearn.metrics import f1_score
    best_t = 0.0
    best_fscore = 0.0
    for t in np.arange(0, 1.1, 0.1):
        pred_labels = [int(s > t) for s in pred_scores]
        fscore = f1_score(gold_scores, pred_labels)
        if (best_fscore < fscore):
            best_fscore = fscore
            best_t = t
    return best_t, best_fscore


def evalFscore(train_gold_scores, train_pred_scores, test_gold_scores, test_pred_scores):
    from  sklearn.metrics import f1_score
    # threshold from train data
    threshold, _ = tuneThreshold(train_gold_scores, train_pred_scores)
    test_pred_labels = [int(s > threshold) for s in test_pred_scores]
    fscore = f1_score(test_gold_scores, test_pred_labels)
    print("fscore: {}".format(fscore))

def evalAccuracy(true_y, predicted_y):
    from sklearn.metrics import accuracy_score, classification_report

    hateful_correct = 0
    total_non_hateful = 0
    non_hateful_correct = 0
    total_hateful = 0

    for values in zip(true_y, predicted_y):
        if values[0] == 0:
            total_non_hateful += 1
            if values[1] == 0:
                non_hateful_correct += 1
        if values[0] == 1:
            total_hateful += 1
            if values[1] == 1:
                hateful_correct += 1


    print(classification_report(true_y, predicted_y, target_names=['Not_Hateful', 'Hateful']))

    print(f"Acc_hate = {hateful_correct} / {total_hateful} = {hateful_correct/total_hateful}")
    print(f"Acc_not_hate = {non_hateful_correct} / {total_non_hateful} = {non_hateful_correct / total_non_hateful}")


    overall_accuracy = accuracy_score(y_true=true_y, y_pred=predicted_y, normalize=True)
    print("accuracy : {}".format(overall_accuracy))
