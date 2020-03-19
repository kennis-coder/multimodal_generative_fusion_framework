# -*- encoding: utf-8 -*-

# @Time    : 11/26/18 10:41 PM
# @Author  : Kennis Yu
# @File    : plot_roc_and_pr.py

from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from numpy import arange, linspace, mean, std, minimum, maximum
from scipy import interp


def plot_roc(results, label):
    tprs = []
    aucs = []
    mean_fpr = linspace(0, 1, 100)

    for (y_true, y_scores) in results:
        y_score = y_scores[:, 1]
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = std(aucs)
    plt.plot(mean_fpr, mean_tpr,
             label=u"Mean ROC of {} \n(AUC = {:.4f} $\pm$ {:.4f})".format(label, mean_auc, std_auc),
             lw=2, alpha=.8)
    std_tpr = std(tprs, axis=0)
    tprs_upper = minimum(mean_tpr + std_tpr, 1)
    tprs_lower = maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.2)


def plot_gfn_rocs(gate_and_interactions, no_gate_and_interactions,
                  gate_and_no_interactions, no_gate_and_no_interactions):
    plot_roc(gate_and_interactions,
             label="gate and interaction")
    plot_roc(no_gate_and_interactions,
             label="no gate and interaction")
    plot_roc(gate_and_no_interactions,
             label="gate and no interaction")
    plot_roc(no_gate_and_no_interactions,
             label="no gate and no interaction")
    plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    plt.xlabel('False positive rate', fontsize=12)
    plt.ylabel('True positive rate', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.show()


def plot_pr(y_true, y_scores, label, fmt):
    y_score = y_scores[:, 1]
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.step(recall, precision, fmt, alpha=0.8, where="post", label=label)
    # plt.fill_between(recall, precision, alpha=0.2)


def plot_gfn_prs(gate_and_interaction, no_gate_and_interaction,
                 gate_and_no_interaction, no_gate_and_no_interaction):
    plot_pr(gate_and_interaction["y_true"], gate_and_interaction["y_scores"],
            label="gate and interaction", fmt='+')
    plot_pr(no_gate_and_interaction["y_true"], no_gate_and_interaction["y_scores"],
            label="no gate and interaction", fmt='x')
    plot_pr(gate_and_no_interaction["y_true"], gate_and_no_interaction["y_scores"],
            label="gate and no interaction", fmt='*')
    plot_pr(no_gate_and_no_interaction["y_true"], no_gate_and_no_interaction["y_scores"],
            label="no gate and no interaction", fmt='.')

    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.show()


def plot_augmented_prs(augmented_by_multimodal, augmented_by_unimodal, imbalanced):
    plot_pr(augmented_by_multimodal["test_target"], augmented_by_multimodal["test_scores"],
            label="augmented data by Multimodal GANs", fmt='b-')
    plot_pr(augmented_by_unimodal["test_target"], augmented_by_unimodal["test_scores"],
            label="augmented data by GANs", fmt='g.')
    plot_pr(imbalanced["test_target"], imbalanced["test_scores"],
            label="imbalanced data", fmt='y*')
    plt.ylim([0.0, 1.0])
    plt.xlim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='best')
    plt.show()


def auto_label(ax, rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                '{:.4f}'.format(height), ha=ha[xpos], va='bottom')


def plot_score_and_time(train_scores, test_scores, elapsed_time):
    ind = arange(len(train_scores))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    rects1 = ax1.bar(ind - width / 2, train_scores, width,
                     color='SkyBlue', hatch='x', label='train scores', alpha=0.8)
    rects2 = ax1.bar(ind + width / 2, test_scores, width,
                     color='orange', hatch='+', label='test scores', alpha=0.8)

    ax1.set_ylim([0.85, 1.])
    ax1.set_xticks(ind)
    ax1.set_xticklabels(train_scores.index)
    ax1.legend()

    # auto_label(ax1, rects1, "left")
    # auto_label(ax1, rects2, "right")

    ax1.set_ylabel('accuracy')
    ax1.grid()

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(ind, elapsed_time, 'b-d')
    ax2.set_ylabel('training minutes')
    plt.show()
