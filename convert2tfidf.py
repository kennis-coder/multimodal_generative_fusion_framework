# -*- encoding: utf-8 -*-

# @Time    : 11/30/18 11:09 AM
# @Author  : Kennis Yu
# @File    : convert2tfidf.py
from tqdm import tqdm
from numpy import concatenate, array
from sklearn.feature_extraction.text import TfidfTransformer

from .utils import write


class CountMatrixizer(object):
    def __init__(self, token_matrix):
        n_row = token_matrix.shape[0]
        token_list = set()

        for i in tqdm(range(n_row), total=n_row, unit="row"):
            token_list |= set(token_matrix[i])

        self.template = dict.fromkeys(token_list, 0)

    def fit(self, token_matrix):
        n_row = token_matrix.shape[0]
        _counts_matrix = []

        for i in tqdm(range(n_row), total=n_row, unit="row"):
            counts = self.template.copy()
            for token in token_matrix[i]:
                counts[token] += 1

            _counts_matrix.append(list(counts.values()))

        _counts_matrix = array(_counts_matrix)
        return _counts_matrix


def convert2tfidf(texts, target, fname):
    cat_texts = concatenate(texts, axis=0)
    count_matrixizer = CountMatrixizer(cat_texts)

    pos_train_counts_matrix = count_matrixizer.fit(texts[0])
    neg_train_counts_matrix = count_matrixizer.fit(texts[1])

    pos_test_counts_matrix = count_matrixizer.fit(texts[2])
    neg_test_counts_matrix = count_matrixizer.fit(texts[3])

    cat_counts_matrix = concatenate((pos_train_counts_matrix,
                                     neg_train_counts_matrix,
                                     pos_test_counts_matrix,
                                     neg_test_counts_matrix),
                                    axis=0)
    target = concatenate(target, axis=0)
    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(cat_counts_matrix, target)
    cat_tfidf_matrix = tfidf_transformer.transform(cat_counts_matrix)
    log_col = array(cat_tfidf_matrix.sum(axis=0) >= 12.05).ravel()

    n_pos_train = pos_train_counts_matrix.shape[0]
    n_neg_train = neg_train_counts_matrix.shape[0]
    n_pos_test = pos_test_counts_matrix.shape[0]
    n_neg_test = neg_test_counts_matrix.shape[0]

    n1 = n_pos_train
    n2 = n_pos_train+n_neg_train
    n3 = n_pos_train+n_neg_train+n_pos_test
    n4 = n_pos_train+n_neg_train+n_pos_test+n_neg_test

    pos_train_tfidf_matrix = cat_tfidf_matrix[0: n1, log_col].toarray()
    neg_train_tfidf_matrix = cat_tfidf_matrix[n1: n2, log_col].toarray()
    pos_test_tfidf_matrix = cat_tfidf_matrix[n2: n3, log_col].toarray()
    neg_test_tfidf_matrix = cat_tfidf_matrix[n3: n4, log_col].toarray()

    tfidf_matrix = (pos_train_tfidf_matrix, neg_train_tfidf_matrix,
                    pos_test_tfidf_matrix, neg_test_tfidf_matrix)
    write(tfidf_matrix, fname)
    return tfidf_matrix