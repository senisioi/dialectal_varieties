import os
import logging
import random
from itertools import product
from collections import defaultdict

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

from tqdm import tqdm


MAX_NO_WD_NGRAMS = 300
MAX_NO_POS_NGRAMS = 2000


logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)


def files_in_folder(mypath):
    return [ os.path.join(mypath,f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath,f)) ]


def read_dir_and_labels_dialects(input_dir, only_these = []):
    train_chunks = []
    train_labels = []
    test_chunks = []
    test_labels = []
    test_documents = []
    all_files = files_in_folder(input_dir)
    all_files.sort()
    for file in files_in_folder(input_dir):
        ending = os.path.basename(file).split('.')[-1]
        dialect = os.path.basename(file).split('.')[0]
        if only_these and dialect not in only_these:
            continue
        if ending == 'train':
            with open(file, 'r', encoding='utf-8') as fin:
                chunks = [text.strip() for text in fin.readlines()]
            train_chunks.extend(chunks)
            train_labels.extend([dialect]*len(chunks))
        elif ending == 'test':
            with open(file, 'r', encoding='utf-8') as fin:
                chunks = [text.strip() for text in fin.readlines()]
            test_chunks.extend(chunks)
            test_labels.extend([dialect]*len(chunks))
            test_documents.append(os.path.basename(file))

    encoder = LabelEncoder()
    encoder.fit(train_labels)
    y = encoder.transform(train_labels)
    y_test = encoder.transform(test_labels)
    return train_chunks, y, test_chunks, y_test, encoder, test_documents


def read_fun(fis_fun):
    fun = set()
    with open(fis_fun, 'r', encoding='utf-8') as fin:
        fun = set([elem.strip() for elem in fin.readlines()])  
    return fun


def log_entropy(matrix):
    if type(matrix) is not np.ndarray:
        matrix = matrix.toarray()
    normalized = matrix / (1 + np.sum(matrix, axis=0))
    nr_docs, _ = matrix.shape
    '''
        g_i = 1 + sum     p_ij * log(p_ij + 1)   
                 j=1,N  ------------------------
                               log(N)                              
    '''
    entropy = 1 + np.sum(np.multiply(normalized, np.log(normalized + 1)), axis=0)/np.log(nr_docs)
    '''
        logent_ij = gi * log(tf_ij + 1)
    '''
    log_ent = entropy * np.log(matrix + 1)
    return log_ent


def run_feature_select(tr_path, fun_path, num_features=20):
    train_chunks, y, test_chunks, y_test, enc, test_docs = read_dir_and_labels_dialects(tr_path)
    print('\t Feature selection: ', tr_path, '\n\t', fun_path)

    model = LogisticRegression(penalty='l2', dual=False, max_iter=10000, tol=0.0001, solver='liblinear',
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)
    fun = read_fun(fun_path)
    cvc = CountVectorizer(max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), vocabulary=fun)
    X = cvc.fit_transform(train_chunks + test_chunks)
    print("log entropy...")
    X = log_entropy(X)
    print("selecting..")
    rfe = RFE(model, n_features_to_select=num_features)
    rfe = rfe.fit(X[:len(train_chunks)], y)
    reverse_vocab = {}
    for k,v in cvc.vocabulary_.items():
        reverse_vocab[v] = k
    print("Features of rank 1:")
    selected = np.where(rfe.ranking_ == 1)[0]
    for s in selected:
        print(reverse_vocab[s])
    print("Features of rank 2:")
    selected = np.where(rfe.ranking_ == 2)[0]
    for s in selected:
        print(reverse_vocab[s])
    result = rfe.predict(X[len(train_chunks):])
    acc_scores = accuracy_score(y_test, result)
    f1_scores = f1_score(y_test, result, average='weighted')  
    C = confusion_matrix(y_test, result)
    C = C / C.astype(float).sum(axis=1) * 100
    print("\tAccuracy", acc_scores)
    print("\tF1 score", f1_scores)
    print("\tConfusion matrix\n", C)
    return rfe, cvc


def custom_lowest_class_cv(X, y, tolerance=10, max_sample=20):
    unq, indices, cnts = np.unique(y, return_counts=True, return_inverse=True)
    nr_of_classes = len(unq)
    smallest_class = cnts.min()
    per_class = {}
    for cls_idx, cls in enumerate(unq):
        per_class[cls] = indices == cls_idx
    per_class_idx = defaultdict(list)
    for cls, data in per_class.items():
        data = np.where(data)[0]
        for start in range(0, len(data), smallest_class):
            end = min([len(data), start + smallest_class])
            if end - start + tolerance < smallest_class:
                continue
            per_class_idx[cls].append((start, end))
    # to enforce order
    ordered_values = []
    for cls in unq:
        ordered_values.append(per_class_idx[cls])
    combinations = list(product(*ordered_values))
    max_sample = min([len(combinations), max_sample])
    class_combinations = random.sample(combinations, max_sample)
    for sample_pos in class_combinations:
        positions = []
        for cls_idx, cls in enumerate(unq):
            start = sample_pos[cls_idx][0]
            end = sample_pos[cls_idx][1]
            positions.extend(np.where(per_class[cls])[0][start:end])
        yield X[positions], y[positions]


def fit_predict_default_model(train_data, train_labels, test_data, vocabulary=None, max_features=None):
    model = LogisticRegression(penalty='l2',
                               dual=False,
                               max_iter=10000,
                               tol=0.0001,
                               solver='liblinear',
                               C=1,
                               fit_intercept=True,
                               intercept_scaling=1.0, 
                               class_weight=None,
                               random_state=1)
    cvc = CountVectorizer(max_features=max_features,
                          strip_accents='unicode',
                          analyzer='word',
                          lowercase=True,
                          token_pattern=r'\w{1,}',
                          ngram_range=(1, 5),
                          vocabulary=vocabulary)
    X = cvc.fit_transform(list(train_data) + list(test_data))
    print("log entropy...")
    X = log_entropy(X)
    model.fit(X[:len(train_data)], train_labels)
    result = model.predict(X[len(train_data):])
    return result


def run_dialect_experiment_multi_sample(tr_path, fun_path=None, max_features=MAX_NO_WD_NGRAMS, only_these = []):
    total_train_chunks, total_y, test_chunks, y_test, enc, test_docs = read_dir_and_labels_dialects(tr_path, only_these=only_these)
    print(enc.classes_)
    if fun_path is None:
        fun = None
    else:
        fun = read_fun(fun_path)
        max_features = None
    print('\t', tr_path, '\n\t', fun_path)
    accs = []
    f1s = []
    confs = []
    idx = 0
    for train_chunks, y in custom_lowest_class_cv(np.array(total_train_chunks), total_y):
        print('Fit-predict on subsample ', idx)
        idx += 1
        result = fit_predict_default_model(train_chunks, y, test_chunks, vocabulary=fun, max_features=max_features)
        acc_sc = accuracy_score(y_test, result)
        f1_sc = f1_score(y_test, result, average='weighted')  
        C = confusion_matrix(y_test, result)
        C = C / C.astype(float).sum(axis=1) * 100
        accs.append(acc_sc)
        f1s.append(f1_sc)
        confs.append(C)
    avg_acc = np.round(np.mean(accs),2)
    avg_f1 = np.round(np.mean(f1s), 2)
    avg_conf = np.round(np.mean(confs, axis=0),2)
    print("\tAccuracy", avg_acc)
    print("\tF1 score", avg_f1)
    print("\tConfusion matrix\n", avg_conf)
    return avg_f1, avg_conf, enc.classes_


def run_kmeans_clust(tr_path, fun_path, lang='en', pos=False):
    from mpl_toolkits.mplot3d import Axes3D 
    data, labels, enc, files = read_dir_and_labels(tr_path)
    fun = read_fun(fun_path)
    print('\t', tr_path, '\n\t', fun_path, '\n\t', lang)
    pca = PCA(n_components=3)
    model = KMeans(n_clusters=3, init='k-means++', random_state=4)
    cvc = CountVectorizer(max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), vocabulary=fun)
    X = cvc.fit_transform(data)
    X = log_entropy(X)
    x_9d = pca.fit_transform(X)
    X_clustered = model.fit_predict(X)
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    etiquet = ['Native', 'Non-native', 'Translated']
    files = np.array(files)    
    fig, ax = plt.subplots()
    for idx,(etiq,color) in enumerate(zip(etiquet, colors)):
        x_vals = x_9d[np.where(labels == idx)]
        ax.scatter(x_vals[:,0], x_vals[:,1], c=color, label=etiq, alpha=0.3, edgecolors='none')
    '''
    3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx,(etiq,color) in enumerate(zip(etiquet, colors)):
        x_vals = x_9d[np.where(labels == idx)]
        ax.scatter(x_vals[:,0], x_vals[:,1],  x_vals[:,2], c=color, label=etiq)#, alpha=0.3, edgecolors='none')
    '''
    ax.legend()
    plt.show()
