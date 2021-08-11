from TriTraining import TriTraining
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import KFold, cross_val_score
import time
from fs_utility import *
from queue import PriorityQueue, Queue


# @ generate classic discernibility matrix for a decision table
def get_discernibility_matrix(label_data, unlabel_data):
    # 有标签部分
    X1 = label_data[:, 0:-1]
    y = label_data[:, -1]
    (n_rows1, m_cols) = X1.shape
    core_flag = np.zeros(m_cols)
    cores = []
    dis_mat = []
    for i in range(n_rows1):
        for j in range(i, n_rows1):
            if y[i] != y[j]:  # 是否标记相同
                xi = X1[i, :]
                xj = X1[j, :]
                e = (xi != xj).astype('int8')  # bool to int
                idx = np.where(e == 1)[0]
                if idx.size > 0:  # there is a case that the same values in condition but different decisions
                    dis_mat.append(e)
                    if idx.size == 1 and core_flag[idx[0]] == 0:
                        cores.append(idx[0])  # core features
                        core_flag[idx[0]] = 1  # 标记为核特征
    if unlabel_data.size == 0:
        dis_mat = np.array(dis_mat)  # list to numpy
        return dis_mat, cores

    # 无标签部分
    X2 = unlabel_data[:, 0:-1]
    (n_rows2, m_cols) = X2.shape
    for i in range(n_rows2):
        for j in range(i + 1, n_rows2):
            xi = X2[i, :]
            xj = X2[j, :]
            e = (xi != xj).astype('int8')  # bool to int
            idx = np.where(e == 1)[0]
            if idx.size > 0:  # there is a case that the same values in condition but different decisions
                dis_mat.append(e)
                if idx.size == 1 and core_flag[idx[0]] == 0:
                    cores.append(idx[0])  # core features
                    core_flag[idx[0]] = 1  # 标记为核特征

    # 有标签和无标签
    for i in range(n_rows1):
        for j in range(n_rows2):
            xi = X1[i, :]
            xj = X2[j, :]
            e = (xi != xj).astype('int8')  # bool to int
            idx = np.where(e == 1)[0]
            if idx.size > 0:  # there is a case that the same values in condition but different decisions
                dis_mat.append(e)
                if idx.size == 1 and core_flag[idx[0]] == 0:
                    cores.append(idx[0])  # core features
                    core_flag[idx[0]] = 1  # 标记为核特征

    dis_mat = np.array(dis_mat)  # list to numpy
    return dis_mat, cores


class Node:
    __slots__ = ('state', 'select_feature', 'gCost', 'hCost')

    def __init__(self, state: list, n_row, n_col, parent=None, feature=None):
        self.state = state  # 存放当前差分矩阵元素项在初始差分矩阵的下标
        if parent is None:
            self.select_feature = []
        else:
            self.select_feature = parent.select_feature.copy()  # 注意要使用深拷贝
            self.select_feature.append(feature)  # 记录当前状态选择的特征
        # gCost和hCost去量纲化
        self.gCost = len(self.select_feature) / n_col  # 已选特征数
        self.hCost = len(self.state) / n_row  # 剩余未区分项数

    def __lt__(self, other):
        return self.gCost + self.hCost < other.gCost + other.hCost


def dm_fs_BeanSearch(label_data, unlabel_data, b_core=False):
    """
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data, guaranteed to be discrete
    y: {numpy array}, shape (n_samples,)
        input class labels
    b_core: {bool}
        add the cores to the reduct or not

    Output
    ------
    F: {numpy array}, shape (n_features,)
        index of selected features, F[0] is the most important feature
    fre_val: {numpy array}, shape: (n_features,)
        corresponding frequency value of selected features

    Reference
    ---------
    Yiyu Yao, et al, Discernibility matrix simplification for constructing attribute reducts,
    Information Sciences, 179 (2009) 867–882.
    """
    # ************01: generate matrix************
    dm, core_feas = get_discernibility_matrix(label_data, unlabel_data)
    # print(dm)

    # ************02: perform absorption law************

    # ************03: add cores to ereduct************
    F = []  # record the selected fatures
    if b_core:
        F = core_feas.copy()
        # remove elements that contain the core features
        for idx in core_feas:
            indices = np.where(dm[:, idx] == 0)[0]
            dm = dm[indices]  # remove the elements of dm that contain the selected features
    if len(dm) == 0:
        return core_feas, core_feas
    ini_node = Node([j for j in range(dm.shape[0])], dm.shape[0], dm.shape[1])  # 初始节点状态
    ini_node.select_feature = F
    open = Queue()  # 采用队列，先进先出
    open.put(ini_node)  # open表放入初始节点
    close = set()  # 哈希去重
    result = []
    while not open.empty() and len(result) != 3:
        cur_node = open.get()
        close.add(str(cur_node.select_feature))
        if len(cur_node.state) == 0:
            result.append(cur_node.select_feature)  # 返回选择的特征、核特征
        # 获取后继节点列表
        sub_nodes = PriorityQueue()  # 采用优先队列，获取最优的前B个后继节点
        # 生成后继节点
        for i in range(dm.shape[1]):
            if cur_node.select_feature and i in cur_node.select_feature: continue
            indices = np.where(dm[cur_node.state, i] == 0)[0]
            child = Node(indices, dm.shape[0], dm.shape[1], parent=cur_node, feature=i)
            sub_nodes.put(child)
        # 遍历后继节点
        B = 0
        while not sub_nodes.empty():
            # while not sub_nodes.empty() and B < 3:
            child = sub_nodes.get()
            if str(child.select_feature) not in close:
                open.put(child)
                B += 1
                if B >= 3: break  # B值取3
        del sub_nodes  # 释放非最优后继节点内存
    return result, core_feas


def mySample(y, train_num, test_num, rnd):
    pos_indices = np.where(y == 1)[0]  # indices of all positive examples, tuple with [0] to array
    neg_indices = np.where(y == 0)[0]  # indices of all negative examples

    train_y1_num = int(len(pos_indices) / len(y) * train_num)  # the number of labeled positive examples
    train_y0_num = train_num - train_y1_num  # the number of labeled negative examples

    test_y1_num = int(len(pos_indices) / len(y) * test_num)  # the number of labeled positive examples
    test_y0_num = test_num - test_y1_num  # the number of labeled negative examples

    # shuffle the indices for initial labeled data
    np.random.seed(rnd)
    np.random.shuffle(pos_indices)
    np.random.shuffle(neg_indices)
    semi_train_index = np.hstack((pos_indices[:train_y1_num], neg_indices[:train_y0_num]))
    semi_test_index = np.hstack((pos_indices[train_y1_num: train_y1_num + test_y1_num],
                                 neg_indices[train_y0_num: train_y0_num + test_y0_num]))

    return semi_train_index, semi_test_index


# @ test the performance of each feature selection method on a given data set
def test_FS():
    # data_name_list = ['arrhythmia', 'credit-a', 'credit-g', 'dermatology', 'lymph', 'vote', 'wbcd', 'wine']
    # data_name_list = ['credit-a', 'vote', 'wine', 'anneal', 'newcylinder-bands',
    # 'newsplice', 'sonar', 'vehicle', 'zalizadeh sani', 'zpolish-companies-bankruptcy-2year',  'zquality-assessment-green',
    # 'spect-test', 'zparkinson-speech-train', 'heart-statlog', 'newcylinder-bands']
    data_name_list = ['credit-a', 'vote', 'anneal', 'newcylinder-bands', 'newsplice', 'vehicle', 'zalizadeh sani']

    # data_name_list = ['credit-a', ]
    for data_name in data_name_list:
        file_path = './data_core/' + data_name + '.data'
        save_path = file_path.replace('.data', '.txt')
        sys.stdout = Logger(save_path)

        # **********01: load data**********
        data = np.loadtxt(file_path, dtype=int, delimiter=',')
        print("\n********************01:loading data*******************")
        print("Data set:", data_name)
        print("Row data shape", data.shape)
        # np.random.shuffle(data)  # 样本顺序随机
        (n_samples, m_features) = data.shape
        m_features = m_features - 1
        for i in np.arange(0.01, 0.055, 0.005):
            print("有标签数据占比：{:.2%}".format(i))
            X = data[:, :-1]
            y = data[:, -1]
            n_train = int(n_samples * i)  # 有标签样本数
            n_test = int(n_samples / 10)
            res1, res2 = [], []
            for j in range(10):
                # 随机抽取数据集，得到traindata, testdata, udata
                train_index, test_index = mySample(y, n_train, n_test, j)
                u_index = list(set(np.arange(data.shape[0])) - set(train_index) - set(test_index))
                traindata = X[train_index]
                trainlabel = y[train_index]
                testdata = X[test_index]
                testlabel = y[test_index]
                udata = X[u_index]

                feas_BeanSearch_withcore, feas_core = dm_fs_BeanSearch(data[train_index], data[u_index], False)
                if len(feas_BeanSearch_withcore) != 3:
                    print(feas_BeanSearch_withcore)
                    print("无法找到三个特征约简子集！")
                    continue
                # print("三个约简特征集：", feas_BeanSearch_withcore)
                # jiaoji = set()
                # for i in feas_BeanSearch_withcore:
                #     jiaoji = jiaoji | set(i)
                # print("约简特征交集：", jiaoji)
                #
                # bingji = set(feas_BeanSearch_withcore[0])
                # for i in feas_BeanSearch_withcore:
                #     bingji = bingji & set(i)
                # print("约简特征并集：", bingji)

                # print("核特征：", sorted(feas_core))
                # print("特征约简率：%.2f%%" % ((1 - len(feas_BeanSearch_withcore[0]) / m_features) * 100))

                ########### cross-validation for performance evaluation  ###########
                clf = KNN(n_neighbors=3)
                clf.fit(traindata, trainlabel)
                res1.append(clf.score(testdata, testlabel))

                TT = TriTraining([KNN(n_neighbors=3), KNN(n_neighbors=3), KNN(n_neighbors=3)])
                TT.fit(feas_BeanSearch_withcore, traindata, trainlabel, udata)
                res2.append(TT.score(testdata, testlabel, feas_BeanSearch_withcore))
            print("单个分类器：", sum(res1) / len(res1))
            print("Tri-Training：", sum(res2) / len(res2))
            print('*' * 50)


if __name__ == '__main__':
    # test feature selection on single data set
    test_FS()
