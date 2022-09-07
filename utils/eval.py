from sklearn.metrics import f1_score, recall_score, precision_score, fbeta_score, hamming_loss, zero_one_loss, precision_score, roc_auc_score, accuracy_score, balanced_accuracy_score

def print_eval_metrics(true_prob_l, pred_prob_l, prt=True):
    pred_l = [x>0.5 for x in pred_prob_l]

    recall = recall_score(true_prob_l, pred_l)
    precision = precision_score(true_prob_l, pred_l)
    f1 = f1_score(true_prob_l, pred_l)
    bac = balanced_accuracy_score(true_prob_l, pred_l)
    acc = accuracy_score(true_prob_l, pred_l)
    try:
        auc = roc_auc_score(true_prob_l, pred_prob_l)
    except ValueError:
        auc = 0.5
    hloss = hamming_loss(true_prob_l, pred_l)
    if prt:
        print("Rec : {:.4f}".format(recall))
        print("Precision : {:.4f}".format(precision))
        print("F1 : {:.4f}".format(f1))
        print("BAC : {:.4f}".format(bac))
        print("Acc : {:.4f}".format(acc))
        print("auc : {:.4f}".format(auc))
        print("hamming loss: {:.4f}".format(hloss))
    return hloss, recall, precision ,f1, bac, acc, auc
