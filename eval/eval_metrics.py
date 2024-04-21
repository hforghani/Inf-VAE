import datetime
import json
import logging
import os
from typing import Tuple

import numpy as np
from sklearn import metrics
from matplotlib import pyplot

from utils import preprocess


def precision_at_k(relevance_score, k):
    """ Precision at K given binary relevance scores. """
    assert k >= 1
    relevance_score = np.asarray(relevance_score)[:k] != 0
    if relevance_score.size != k:
        raise ValueError('Relevance score length < K')
    return np.mean(relevance_score)


def recall_at_k(relevance_score, k, m):
    """ Recall at K given binary relevance scores. """
    assert k >= 1
    relevance_score = np.asarray(relevance_score)[:k] != 0
    if relevance_score.size != k:
        raise ValueError('Relevance score length < K')
    return np.sum(relevance_score) / float(m)


def fpr_at_k(relevance_score, k, ir_count
             # , in_count, inp, out, target, m
             ):
    """ False positive rate at K given binary relevance scores. """
    assert k >= 1
    relevance_score = np.asarray(relevance_score)[:k] != 0
    if relevance_score.size != k:
        raise ValueError('Relevance score length < K')
    # fp = len(set(out[:k]) - set(target))
    fp = relevance_score.size - np.sum(relevance_score)

    # logging.info(f"k = {k}\n"
    #              f"in_count = {in_count}\n"
    #              f"input = {inp}\n"
    #              f"output = {out}\n"
    #              f"target = {target}\n"
    #              f"relevant count = {m}\n"
    #              f"relevance_score = {''.join('1' if r else '0' for r in relevance_score)}\n"
    #              f"relevance_score.size = {relevance_score.size}\n"
    #              f"np.sum(relevance_score) = {np.sum(relevance_score)}\n"
    #              f"fp1 = {relevance_score.size - np.sum(relevance_score)}\n"
    #              f"fp2 = {fp}\n"
    #              f"ir_count = {ir_count}\n")

    assert fp <= ir_count

    return fp / ir_count


def f1_at_k(relevance_score, k, m):
    assert k >= 1
    relevance_score = np.asarray(relevance_score)[:k] != 0
    if relevance_score.size != k:
        raise ValueError('Relevance score length < K')
    precision = np.mean(relevance_score)
    recall = np.sum(relevance_score) / float(m)
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    # logging.info(f"relevance_score = {''.join('1' if r else '0' for r in relevance_score)}\n"
    #              f"k = {k}, precision = {precision}, recall = {recall}, f1 = {f1}")
    return f1


def mean_precision_at_k(relevance_scores, k):
    """ Mean Precision at K given binary relevance scores. """
    mean_p_at_k = np.mean(
        [precision_at_k(r, k) for r in relevance_scores]).astype(np.float32)
    return mean_p_at_k


def mean_recall_at_k(relevance_scores, k, m_list):
    """ Mean Recall at K:  m_list is a list containing # relevant target entities for each data point. """
    mean_r_at_k = np.mean([recall_at_k(r, k, M) for r, M in zip(relevance_scores, m_list)]).astype(np.float32)
    return mean_r_at_k


def mean_fpr_at_k(relevance_scores, k, ir_list
                  # , in_counts, inputs, outputs, targets, m_list
                  ):
    """ Mean Recall at K:  ir_list is a list containing # irrelevant candidate entities for each data point. """
    mean_fpr = np.mean(
        [fpr_at_k(r, k, ir_count,
                  # in_count, inp, out, target, m
                  ) for r, ir_count
         # , in_count, inp, out, target, m
         in zip(relevance_scores, ir_list
                # , in_counts, inputs, outputs, targets, m_list
                )]).astype(np.float32)
    return mean_fpr


def mean_f1_at_k(relevance_scores, k, m_list):
    # logging.info(f"m = {m_list}")
    mean_f1 = np.mean([f1_at_k(r, k, M) for r, M in zip(relevance_scores, m_list)]).astype(np.float32)
    return mean_f1


def max_f1(relevance_scores, k_list, m_list):
    f1_values = [mean_f1_at_k(relevance_scores, k, m_list) for k in k_list]
    # logging.info(f"f1_values = {f1_values}")
    max_val = np.max(f1_values)
    return max_val


def average_precision(relevance_score, K, m):
    """ For average precision, we use K as input since the number of prediction targets is not fixed
    unlike standard IR evaluation. """
    r = np.asarray(relevance_score) != 0
    out = [precision_at_k(r, k + 1) for k in range(0, K) if r[k]]
    if not out:
        return 0.
    return np.sum(out) / float(min(K, m))


def MAP(relevance_scores, k, m_list):
    """ Mean Average Precision -- MAP. """
    map_val = np.mean([average_precision(r, k, M) for r, M in zip(relevance_scores, m_list)]).astype(np.float32)
    return map_val


def MRR(relevance_scores):
    """ Mean reciprocal rank -- MRR. """
    rs = (np.asarray(r).nonzero()[0] for r in relevance_scores)
    mrr_val = np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs]).astype(np.float32)
    return mrr_val


def get_masks(top_k, inputs):
    """ Mask the dummy sequences  -- : 0 if .. 1 if seed set is of size > 1. """
    masks = []
    for i in range(0, top_k.shape[0]):
        seeds = set(inputs[i])
        if len(seeds) == 1 and list(seeds)[0] == preprocess.start_token:
            masks.append(0)
        else:
            masks.append(1)
    return np.array(masks).astype(np.int32)


def remove_seeds(top_k, inputs):
    """ Replace seed users from top-k predictions with -1. """
    result = []
    for i in range(0, top_k.shape[0]):
        seeds = set(inputs[i])
        lst = list(top_k[i])  # top-k predicted users.
        logging.info(f"inputs[{i}] = {inputs[i]}\n"
                     f"top_k[{i}] = {top_k[i]}")
        for s in seeds:
            if s in lst:
                lst.remove(s)
        for k in range(len(top_k[i]) - len(lst)):
            lst.append(-1)
        result.append(lst)
    return np.array(result).astype(np.int32)


def get_relevance_scores(top_k_filter, targets):
    """ Create binary relevance scores by checking if the top-k predicted users are in target set. """
    output = []
    # logging.info(f"targets = {targets}")
    for i in range(0, top_k_filter.shape[0]):
        z = np.isin(top_k_filter[i], targets[i])
        output.append(z)
    return np.array(output)


def auc_roc(fprs: list, tprs: list):
    """ area under curve of ROC """
    fprs, tprs = prepare_roc(fprs, tprs)
    return metrics.auc(fprs, tprs)


def prepare_roc(fprs, tprs) -> Tuple[np.array, np.array]:
    """ Preprocess fpr and tpr values and sort them to calculate auc_roc or to plot ROC """
    # Every ROC curve must have 2 points <0,0> (no output) and <1,1> (returning all reference set as output).
    if 0 not in fprs:
        fprs = [0] + fprs
        tprs = [0] + tprs
    if 1 not in fprs:
        fprs.append(1)
        tprs.append(1)
    fprs, tprs = np.array(fprs), np.array(tprs)
    indexes = fprs.argsort()
    fprs = fprs[indexes]
    tprs = tprs[indexes]
    return fprs, tprs


def save_roc(fpr_list: list, tpr_list: list, dataset: str):
    """
    Save ROC plot as png and FPR-TPR values as json.
    """
    fpr, tpr = prepare_roc(fpr_list, tpr_list)
    pyplot.figure()
    pyplot.plot(fpr, tpr)
    pyplot.axis((0, 1, 0, 1))
    pyplot.xlabel("fpr")
    pyplot.ylabel("tpr")
    results_path = 'results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    base_name = f'{dataset}-roc-{datetime.datetime.now()}'.replace(" ", "-")
    pyplot.savefig(os.path.join(results_path, f'{base_name}.png'))
    # pyplot.show()
    with open(os.path.join(results_path, f'{base_name}.json'), "w") as f:
        json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, f)
