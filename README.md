# 一、论文思路
## 1.差分矩阵的构建
样本数据分为有标签数据和无标签数据。构建差分矩阵分为三个部分：有标签和有标签，有标签和无标签，无标签和无标签。在构造差分矩阵时，每一个无标签数据都看作是一个新的类。
## 2.束搜索
束搜索还需要想出更好的hCost，在满足可纳性同时，计算的时间复杂不能太高。束搜搜的大致流程为：每层派出三个分支，当特征子集能区分所有的差分项，记录其为搜索结果，最后得到三个约减的特征子集。示意图如下：
![图片1](https://user-images.githubusercontent.com/59695008/129035127-28778e64-369d-4f1f-9684-0bc39a4da139.png)

## 3.集成学习
对获取的三个特征约简子集选取适当地分类器进行分类，用投票地方式确定样本最后地分类结果。记录选取的核特征，交集，并集，下标，个数。选合二类或类别较少的数据集，选取12个到16个集成效果较好的数据集；分析三组特征多样性（交并），尝试改成启发式信息，提升特征子集的差异性。


## 4.迁移到Tri-trianing
研究Tri-training的代码，尝试利用在L上训练的三个特征子集分类集，进一步在U上迭代半监督学习。

# 二、论文进展
## 1.差分矩阵的构建和束搜索
完成了半监督的差分矩阵的构建：有标记和有标记，无标记和无标记，有标记和无标记 三类进行差分。除此之外，实现了用束搜索对特征的选取：设计启发式信息（尝试了贪心，效果没有hCost+gCost的好，目前的还是已选特征数+剩余差分项的启发式信息），无标签数据按照总体样本的[0.05:0.55:0.05]比例，得出三个约简特征子集，并计算出三个特征子集的交集，并集，约简率。
结果说明：
* 我尝试的是加入核特征之后继续选择特征，有的数据集由于本身的特征数就不多，而且还有大部分无标签数据，因此有出现核特征数即为总体特征数的情况。对于这种情况，本实验不作考虑。
* 属性约简率在不同数据集上的测试结果大致为：随着有标签样本占比增大呈现先递减后递增趋势，变化的幅度微小。猜测可能原因是：有标签样本占比较小时，无标签样本之间的差本项数量大；有标签样本占比较大时，有标签与无标签之间的差分项数量大。

## 2.集成学习：
实验使用四种分类器：KNN，SVM，Bayes，DecisionTree，采用十折交叉验证的方式进行评估。这部分我自己写了一个交叉验证函数myCrossValidScore：分别在三组属性约简后得到的样本子集上使用四种分类器测试，并将三组预测结果采用集成投票的方式得到一组新的预测结果，分别计算其预测的准确度。
结果说明：
由于用于集成的样本组数较少（只有三组），存在多分类的数据集，而且有的分类器原本的预测准确度较差，集成的精确度并不总能优于单独训练的结果。
最后，我选出了15组实验数据：'credit-a', 'vote', 'wine', 'anneal', 'mfeat-fourier', 'mfeat-karhunen', 'newcylinder-bands', 'newsplice', 'sonar', 'vehicle', 'zalizadeh sani', 'zpolish-companies-bankruptcy-2year',  'zquality-assessment-green',
'spect-test', 'zparkinson-speech-train', 'heart-statlog', 'newcylinder-bands'。等等。这些数据集都是二分类或分类数较少，在这些数据集上运行集成学习的算法，集成后的准确性往往要比三个单独训练特征约简子集的准确性高（结果不总是这样，在某一带标签数据比例下，某个分类器集成后准确性提升不明显，但这是少数情况）。

## 3.迁移到Tri-trianing
认真阅读周志华的tri-training介绍的论文后，我将tritriang的伪代码修改如下：
"""
tri-training(L, U, clf, Selectefeatures)
  Input: L: Original labeled example set
         U: Unlabeled example set
         clf: classify algorithm
         Selectfeatures: 属性约简后的三个特征子集在原数据集的下标
      for i ∈ {1...3}: do
          clf.fit(L_X[:, feas[i]], L_y)   # 将BootstrapSample改为三个不同的特征约简子集
  e_prime[i] = 0.5
          l_prime[i] = 0
  repeat until none of clf[i](i∈{1..3}) changes:
  for i ∈{1..3} do:
      update[i] = False
  Li_X = ∅ # to save proxy labeled data 
              e[i] = MeasureError(L.X, L.y, j, k, Selectfeatures)
              if e[i] < e_prime[i]:
  then for every x ∈ U do:
  if clf[j](x[Selectfeasures[j]) == clf[k](x[Selectfeasures[k]) (j, k ≠ i)
  then Li_X ← Li_X ∪ {x, h[j](x[Selectfeasures[j])}
                  end of for
  if l_prime[i] == 0:  # no updated before
                      l_prime[i] = int(e[i] / (e_prime[i] - e[i]) + 1)
                  if l_prime[i] < len(Li_y[i]):
                      if e[i] * len(Li_y[i]) < e_prime[i] * l_prime[i]:
                          update[i] = True
                      else if l_prime[i] > e[i] / (e_prime[i] - e[i]):
                          L_index = Subsample(Li_X, int(e_prime[i] * l_prime[i] / e[i] - 1))
                          update[i] = True

          for i ∈{1..3} do:
              if update[i] == TRUE:
                  clf[i].fit( (L_X ∪Li_X[i])[feas[i]], (L_y ∪ Li_y[i])[feas[i]])
                  e_prime[i] = e[i]
                  l_prime[i] = len(Li_y[i])
      end of repeat
  Output:  clf(x) ← arg max∑1 {y∈label, i: clf[i](x) == y}

  def MeasureError(X, y, j, k, feas):
      j_pred = clf[j].predict(X[feas[j]])
      k_pred = clf[k].predict(X[feas[k]])
      wrong_index = np.logical_and(j_pred != y, k_pred == j_pred)
      return sum(wrong_index) / sum(j_pred == k_pred)


  def mySample(y, train_num, test_num, rnd):
  “”””
  train_num：训练集样本数
  test_num：测试集样本数
  rnd：随机数种子，总共进行10轮抽样，每轮的随机数种子互异
  “”””
      pos_indices =  (y == 1).indices   # indices of all positive examples
  neg_indices = (y == 0).indices  # indices of all negative examples

      train_y1_num = int(len(pos_indices) / len(y) * train_num)  # the number of labeled positive examples
      train_y0_num = train_num - train_y1_num  # the number of labeled negative examples

      test_y1_num = int(len(pos_indices) / len(y) * test_num)  # the number of labeled positive examples
      test_y0_num = test_num - test_y1_num  # the number of labeled negative examples

      # shuffle the indices for initial labeled data
      np.random.seed(rnd)
      np.random.shuffle(pos_indices)
      np.random.shuffle(neg_indices)

  # 返回训练集下标
      semi_train_index=np.hstack((pos_indices[:train_y1_num],
  neg_indices[:train_y0_num]))
  # 返回测试集下标
      semi_test_index= np.hstack((pos_indices[train_y1_num: train_y1_num + test_y1_num],                               neg_indices[train_y0_num: train_y0_num + test_y0_num]))
  # 返回无标签样本下标
      unlabel_index = set(np.arrange(len(y)) - set(semi_test_index) - set(semi_train_index)

      return semi_train_index, semi_test_index, unlabel_index
"""
有该改动的代码用红色字体标号。改动的内容主要为：将初始的bootstrap sampling改为三个特征约简后的子集。每个分类器的训练的数据都是基于对应的特征约简后的子集。训练集、测试集、无标签数据样本的选取见mySample()函数，函数主要思想为：确保训练集，测试集中正负样本比例与原始数据集一致，剩余的样本均视为无标签样本
实验设置如下：
分类器clf设置为KNN，数据集从集成效果较好的15组数据中选取，以0.005为步长，从0.005-0.05设置有标签数据比例。traindata为原始数据中有标签的数据，testdata是另外的10%的原始数据，udata是剩下的无标签数据。设置两组对照实验：一组使用单分类器clf（这里使用的是KNN）在traindata上训练，另外一组使用tri-traing(三个clf)在属性约简后的traindata和udata上训练（属性约简子集通过束搜索对triandata和udata搜索得到）。每个有标签数据比例下进行10轮随机抽样测试，最后求出两组对照实验10次预测准确度的均值。
程序输出如下：
[程序输出.txt](https://github.com/Luo-ziming/Semi-Supervised-Learning/files/6968680/default.txt)
最后试验结果总结如下：
结果说明：
1.虽然保持了训练集、测试集的分布与原始数据集一致，并且进行了10次不同的随机抽样，但tri-training对于分类性能的提升仍然不太稳定。例如：”anneal”，”vote”， “credit-a”等数据集的运行结果。
2.每个有标签数据比例下都要进行10组抽样，程序的时间复杂度极高。



参考文献：
[1] Z.-H. Zhou and M. Li. Tri-training: exploiting unlabeled data using three classifiers. IEEE Transactions on Knowledge and Data Engineering. 2005, vol.17, no.11, pp.1529-1541. 
