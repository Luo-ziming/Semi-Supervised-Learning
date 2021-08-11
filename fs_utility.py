from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import sys


# @ redirect the console message to file
class Logger(object):
    def __init__(self, file_name="Default.log"):
        self.terminal = sys.stdout
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# @ merge a matrix with discrete variables in column to a new discrete variable
def merge_matrix(mat_X):
    assert mat_X.size != 0

    new_x, idx = np.unique(mat_X, axis=0, return_index=True)
    n_es = new_x.shape[0]
    n_fs = new_x.shape[1]
    new_vec = np.array(
        [np.where((new_x == x.repeat(n_es).reshape(-1, n_es).T).sum(axis=1) == n_fs)[0] for x in mat_X]).flatten()

    return new_vec


# @ merge two discrete variables to a new discrete variable
def merge_two_variables(a, b):
    assert a.shape == b.shape

    c = np.vstack((a, b)).T   # row vectors stack, convert to column vectors
    unique_c, idx = np.unique(c, axis=0, return_index=True)
    new_vec = np.array([np.where((x[0] == unique_c[:, 0]) & (x[1] == unique_c[:, 1])) for x in c]).flatten()

    return new_vec


# @ determine proxy label for unlabeled data according to class prior and class ratio of labeled data
def proxy_label_fun(pr_u=0.2, n_all_examples=1000, alpha=0.1, beta=0.5, gamma=500, epsilon=0.0002):
    # prior part
    scale_weight = pow(1 + epsilon, n_all_examples)
    if pr_u <= 0.5:
        pr_prior = min(pr_u * scale_weight, 0.5)
    else:
        pr_prior = 1 - min((1 - pr_u) * scale_weight, 0.5)

    # initial part
    pr_init = pow(beta, 1 + pow(np.e, -gamma * epsilon * alpha * n_all_examples))

    # determination of proxy label
    pr_all = pr_init * pr_prior
    if pr_all > 0.5:
        y_proxy = 0
    elif pr_all < 0.5:
        y_proxy = 1
    else:
        y_proxy = int(pr_u <= 0.5)
    print(pr_prior, pr_init, pr_all, y_proxy)

    return y_proxy


# @ Tests the relationship of two sets
def sets_relation(list_a, list_b):
    """
    L1 = list([197, 233, 149, 167,  245, 33, 110, 0])
    L2 = list([197, 233, 14, 29, 210, 140, 3, 0])
    b_equal, inter, union, diff = sets_relation(L1, L2)
    print(sets_relation(L1, L2))
    """

    b_equal = 0

    set_a = set(list_a)
    set_b = set(list_b)

    set_inter = set_a.intersection(set_b)
    set_union = set_a.union(set_b)
    set_diff = set_a.difference(set_b)

    if len(set_union) == len(set_a):
        b_equal = 1

    return b_equal, list(set_inter), list(set_union), list(set_diff)


# @ perform the accuracy evaluation of feature subsets, including two subsets and all features
def test_feature_subsets_performance(name_data, subset1, subset2):
    # load data
    data = np.loadtxt(name_data, dtype=int, delimiter=',')
    (n_samples, m_features) = data.shape
    print(name_data, data.shape)

    # cross-validation for performance evaluation**********
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf = svm.LinearSVC(max_iter=100000)  # linear kernel SVM
    clf = svm.SVC()

    acc_m1 = np.array([0.0] * 10)
    acc_m2 = np.array([0.0] * 10)
    acc_all = np.array([0.0] * 10)

    for rnd in range(0, 10):
        # shuffle the examples
        np.random.seed(rnd)
        np.random.shuffle(data)
        print("\nRandom: ", rnd)

        X = data[:, 0:m_features - 1]
        y = data[:, -1].copy()
        print("01: loading data: ", data.shape)

        # **********03: performance evaluation**********
        reduced_X = X[:, subset1]  # obtain the dataset on the selected features
        scores_m1 = cross_val_score(clf, reduced_X, y, cv=kf)
        acc_m1[rnd] = np.mean(scores_m1)
        # print(scores_m1)
        # print("Accuracy m1 : %0.4f (+/- %0.4f)" % (scores_m1.mean(), scores_m1.std() * 2))

        reduced_X = X[:, subset2]  # obtain the dataset on the selected features
        scores_m2 = cross_val_score(clf, reduced_X, y, cv=kf)
        acc_m2[rnd] = np.mean(scores_m2)
        # print(scores_m2)
        # print("Accuracy m2 : %0.4f (+/- %0.4f)" % (scores_m2.mean(), scores_m2.std() * 2))

        scores_all = cross_val_score(clf, X, y, cv=kf)
        acc_all[rnd] = np.mean(scores_all)
        # print(scores_all)
        # print("Accuracy all : %0.4f (+/- %0.4f)" % (scores_all.mean(), scores_all.std() * 2))

    print("Accuracy m1 avg: ", np.mean(acc_m1), ", std: ", np.std(acc_m1),
          "Accuracy m2 avg: ", np.mean(acc_m2), ", std: ", np.std(acc_m2),
          "Accuracy all avg: ", np.mean(acc_all), ", std: ", np.std(acc_all))

    return [np.mean(acc_m1), np.std(acc_m1), np.mean(acc_m2), np.std(acc_m2), np.mean(acc_all), np.std(acc_all)]


# @ perform the accuracy evaluation of feature subsets after backward removing feature
def test_feature_subsets_remove_performance(name_data, subset_feas, acc_all):
    # load data
    data = np.loadtxt(name_data, dtype=int, delimiter=',')
    (n_samples, m_features) = data.shape
    print(name_data, data.shape)

    # cross-validation for performance evaluation**********
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    clf1 = KNeighborsClassifier(n_neighbors=3)
    clf2 = svm.SVC()

    # save experimental results to csv
    cols_name = ['name', 'n_samples', 'm_features', 'feature_subset', 'mde_value',
                 'mean_knn', 'std_knn', 'all_scores_knn', 'mean_svm', 'std_svm', 'all_scores_svm',
                 'indices_num', 'indices_rate', 'noise_rate']
    df_acc = pd.DataFrame([], columns=cols_name)

    acc_m1 = np.array([0.0] * 10)
    acc_m2 = np.array([0.0] * 10)

    # record performance with all features
    # all features to one vector
    vector_all = merge_matrix(data[:, 0:m_features - 1])

    # calculate entropy between all features and target class
    all_mde, all_indices = maximum_decision_entropy(vector_all, data[:, -1], n_samples)
    acc_rate = len(all_indices) / n_samples
    noise_rate = 1 - acc_rate

    for rnd in range(0, 10):
        # shuffle the examples
        np.random.seed(rnd)
        np.random.shuffle(data)
        print("\nRandom: ", rnd)

        X = data[:, 0:m_features - 1]
        y = data[:, -1].copy()

        scores_all = cross_val_score(clf1, X, y, cv=kf)
        acc_m1[rnd] = np.mean(scores_all)

        scores_all = cross_val_score(clf2, X, y, cv=kf)
        acc_m2[rnd] = np.mean(scores_all)

    idx = 0
    df_acc.loc[idx] = [name_data, n_samples, m_features, [m_features, m_features], all_mde,
                       np.mean(acc_m1), np.std(acc_m1), acc_m1, np.mean(acc_m2), np.std(acc_m2), acc_m2,
                       len(all_indices), acc_rate, noise_rate]
    idx = idx + 1

    # record performance with reduced features
    while len(subset_feas) >= 1:

        X = data[:, 0:m_features - 1]
        y = data[:, -1].copy()

        # all features to one vector
        vector_all = merge_matrix(X[:, subset_feas])

        # calculate entropy between all features and target class
        all_mde, curr_indices = maximum_decision_entropy(vector_all, y, n_samples)
        acc_rate = len(curr_indices) / n_samples
        noise_rate = 1 - acc_rate

        acc_m1 = np.array([0.0] * 10)
        acc_m2 = np.array([0.0] * 10)

        for rnd in range(0, 10):
            # shuffle the examples
            np.random.seed(rnd)
            np.random.shuffle(data)
            print("\nRandom: ", rnd)

            X = data[:, 0:m_features - 1]
            y = data[:, -1].copy()
            print("01: loading data: ", data.shape)

            # **********03: performance evaluation**********
            reduced_X = X[:, subset_feas]  # obtain the dataset on the selected features
            scores_all = cross_val_score(clf1, reduced_X, y, cv=kf)
            acc_m1[rnd] = np.mean(scores_all)

            scores_all = cross_val_score(clf2, reduced_X, y, cv=kf)
            acc_m2[rnd] = np.mean(scores_all)

        df_acc.loc[idx] = [name_data, n_samples, m_features, subset_feas.copy(), all_mde,
                           np.mean(acc_m1), np.std(acc_m1), acc_m1, np.mean(acc_m2), np.std(acc_m2), acc_m2,
                           len(all_indices) - len(curr_indices), acc_rate, noise_rate]
        idx = idx + 1
        subset_feas.pop()

    save_name = name_data.replace('.data', '.csv')
    df_acc.to_csv(save_name, index=False, sep=',')


def read_excel_data():
    file_name = "./mde_performance_all_parameters_selected.xlsx"
    df_data = pd.read_excel(file_name)
    n_rows, m_cols = df_data.shape

    for i in range(14, 18):
        data_name = df_data.iloc[i, 0]  # str to list

        """
        test whether two feature subsets are equal
        feas_subset1 = eval(df_data.iloc[i, 4])  # str to list
        feas_subset2 = eval(df_data.iloc[i, 5])  # str to list
        print(i, data_name, feas_subset1, feas_subset2)
        b_flag, _, _, _ = sets_relation(feas_subset1, feas_subset2)
        df_data.iloc[i, 16] = b_flag
        print(i, feas_subset1, feas_subset2, b_flag)
        """

        """
        test feature subsets performance
        all_ret = test_feature_subsets_performance("./data_core/" + data_name, feas_subset1, feas_subset2)
        print(all_ret)
        df_data.iloc[i, 12:18] = all_ret

        save_name = file_name.replace('.xlsx', 'results.csv')
        df_data.to_csv(save_name)
        """

        """
        test the performance of the feature subsets after back-ward removal 
        all_acc = []
        feas_subset = eval(df_data.iloc[i, 4])  # str to list
        all_acc.append(df_data.iloc[i, 10])
        all_acc.append(df_data.iloc[i, 11])
        all_acc.append(df_data.iloc[i, 16])
        all_acc.append(df_data.iloc[i, 17])
        test_feature_subsets_remove_performance("./data_core/" + data_name, feas_subset, all_acc)
        print(data_name, feas_subset, all_acc)
        """

        col_idx = 22
        save_idx = 64
        while col_idx < 24:
            feas_subset1 = eval(df_data.iloc[i, col_idx])  # str to list
            feas_subset2 = eval(df_data.iloc[i, col_idx + 1])  # str to list
            print(i, col_idx, data_name, feas_subset1, feas_subset2)
            all_ret = test_feature_subsets_performance("./data_core/" + data_name, feas_subset1, feas_subset2)
            df_data.iloc[i, save_idx:save_idx + 4] = all_ret[0:4]
            col_idx = col_idx + 2
            save_idx = save_idx + 4

        df_data.iloc[i, save_idx] = all_ret[4]
        df_data.iloc[i, save_idx + 1] = all_ret[5]
        save_name = file_name.replace('.xlsx', 'results.csv')
        df_data.to_csv(save_name)


if __name__ == '__main__':
    # read_excel_data()

    # test set_relation
    L1 = list([9, 8, 2, 7, 3, 4])
    L2 = list([9, 12, 6, 3, 4])
    b_equal, inter, union, diff = sets_relation(L1, L2)
    L1.sort()
    L2.sort()
    print(len(L1), len(L2), L1, L2, len(inter))
