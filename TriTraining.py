import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class TriTraining:
    def __init__(self, classifier=RandomForestClassifier()):
        if sklearn.base.is_classifier(classifier):
            self.classifiers = [sklearn.base.clone(classifier) for i in range(3)]
        else:
            self.classifiers = [sklearn.base.clone(classifier[i]) for i in range(3)]

    def fit(self, feas, L_X, L_y, U_X):
        for i in range(3):
            self.classifiers[i].fit(L_X[:, feas[i]], L_y)
        e_prime = [0.5] * 3
        l_prime = [0] * 3
        e = [0] * 3
        update = [False] * 3
        Li_X, Li_y = [[]] * 3, [[]] * 3  # to save proxy labeled data
        improve = True
        self.iter = 0

        while improve:
            self.iter += 1  # count iterations

            for i in range(3):
                j, k = np.delete(np.array([0, 1, 2]), i)
                update[i] = False
                e[i] = self.measure_error(L_X, L_y, j, k, feas)
                if e[i] < e_prime[i]:
                    U_y_j = self.classifiers[j].predict(U_X[:, feas[j]])
                    U_y_k = self.classifiers[k].predict(U_X[:, feas[k]])
                    Li_X[i] = U_X[U_y_j == U_y_k]  # when two models agree on the label, save it
                    Li_y[i] = U_y_j[U_y_j == U_y_k]
                    # print(l_prime)
                    if l_prime[i] == 0:  # no updated before
                        l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
                    if l_prime[i] < len(Li_y[i]):
                        if e[i] * len(Li_y[i]) < e_prime[i] * l_prime[i]:
                            update[i] = True
                        elif l_prime[i] > e[i] / (e_prime[i] - e[i]):
                            L_index = np.random.choice(len(Li_y[i]), int(e_prime[i] * l_prime[i] / e[i] - 1))
                            Li_X[i], Li_y[i] = Li_X[i][L_index], Li_y[i][L_index]
                            update[i] = True

            for i in range(3):
                if update[i]:
                    self.classifiers[i].fit(np.append(L_X, Li_X[i], axis=0)[:, feas[i]], np.append(L_y, Li_y[i], axis=0))
                    e_prime[i] = e[i]
                    l_prime[i] = len(Li_y[i])

            if update == [False] * 3:
                improve = False  # if no classifier was updated, no improvement

    def predict(self, X, feas):
        pred = np.asarray([self.classifiers[i].predict(X[:, feas[i]]) for i in range(3)])
        pred[0][pred[1] == pred[2]] = pred[1][pred[1] == pred[2]]
        return pred[0]

    def score(self, X, y, feas):
        return sklearn.metrics.accuracy_score(y, self.predict(X, feas))

    def measure_error(self, X, y, j, k, feas):
        """
        返回两个分类器的预测的结果：预测结果相同且错误数 / 预测相结果相同数
        """
        j_pred = self.classifiers[j].predict(X[:, feas[j]])
        k_pred = self.classifiers[k].predict(X[:, feas[k]])
        wrong_index = np.logical_and(j_pred != y, k_pred == j_pred)
        # wrong_index =np.logical_and(j_pred != y_test, k_pred!=y_test)
        return sum(wrong_index) / sum(j_pred == k_pred)


