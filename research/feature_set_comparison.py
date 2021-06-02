import scipy
from scipy.sparse import coo_matrix, hstack
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from network_feature_generation import *
from utils import *


def train_network_feature_classifier(all_bills):
    base_model = LogisticRegression(random_state=1, max_iter=5000, n_jobs=-1, penalty='l2', C=10,
                                    class_weight='balanced')
    bill_id_to_data = get_network_data()
    all_bills = create_network(all_bills, bill_id_to_data)
    X = all_bills[['text', 'centrality',
                   'closeness', 'clustering',
                   'page_rank', 'betweenness',
                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15]]
    y = all_bills['lobbied']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    X_train_network = X_train[['centrality', 'closeness',
                               'clustering', 'page_rank',
                               'betweenness', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                               11, 12, 13, 14, 15]]
    X_test_network = X_test[['centrality', 'closeness',
                             'clustering', 'page_rank',
                             'betweenness', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                             11, 12, 13, 14, 15]]
    base_model.fit(X_train_network, y_train)
    y_pred_proba = base_model.predict_proba(X_test_network)[:, 1]
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    print('auc', auc)
    y_pred = base_model.predict(X_test_network)
    f1 = metrics.f1_score(y_test, y_pred)
    print('f1: ', f1)

def train_textual_feature_classifier(all_bills):
    base_model = LogisticRegression(random_state=1, max_iter=5000, n_jobs=-1, penalty='l2', C=10,
                                    class_weight='balanced')
    X = all_bills[['b_id', 'text']]
    y = all_bills[['lobbied']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    X_train_tf, X_test_tf, vectoried = tfidf_features(X_train.text, X_test.text, ngram_range=(1, 2))
    base_model.fit(X_train_tf, y_train)
    y_pred_proba = base_model.predict_proba(X_test_tf)[:, 1]
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    print('auc', auc)
    y_pred = base_model.predict(X_test_tf)
    f1 = metrics.f1_score(y_test, y_pred)
    print('f1: ', f1)


def train_textual_and_network_feature_classifier(all_bills):
    base_model = LogisticRegression(random_state=1, max_iter=5000, n_jobs=-1, penalty='l2', C=10,
                                    class_weight='balanced')
    bill_id_to_data = get_network_data()
    all_bills = create_network(all_bills, bill_id_to_data)

    X = all_bills[['b_id', 'text']]
    y = all_bills[['lobbied']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    X_train_tf, X_test_tf, vectoried = tfidf_features(X_train.text, X_test.text, ngram_range=(1, 2))

    X = all_bills[['text', 'centrality',
                   'closeness', 'clustering',
                   'page_rank', 'betweenness',
                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15]]
    y = all_bills['lobbied']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    X_train_network = X_train[['centrality', 'closeness',
                               'clustering', 'page_rank',
                               'betweenness', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                               11, 12, 13, 14, 15]]
    X_test_network = X_test[['centrality', 'closeness',
                             'clustering', 'page_rank',
                             'betweenness', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                             11, 12, 13, 14, 15]]
    X_train_network_sparse = scipy.sparse.csr_matrix(X_train_network.values)
    X_test_network_sparse = scipy.sparse.csr_matrix(X_test_network.values)
    X_train_full = hstack([X_train_tf, X_train_network_sparse])
    X_test_full = hstack([X_test_tf, X_test_network_sparse])

    base_model.fit(X_train_full, y_train)
    y_pred_proba = base_model.predict_proba(X_test_full)[:, 1]
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    print('auc', auc)
    y_pred = base_model.predict(X_test_full)
    f1 = metrics.f1_score(y_test, y_pred)
    print('f1: ', f1)



