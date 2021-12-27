#coding=utf-8
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
import pickle






train_data = pd.read_csv(".\\data\\dt_train_data.csv")

# 分析数据集信息
# print("*" * 30 + " info " + "*" * 30)
# print(train_data.info())
# print(train_data.head())

# `survived`字段表示是否生存，以此作为预测目标
y = train_data['sat']
y = y[:5000]
# print("*" * 30 + " y " + "*" * 30)
# print(y.head())

# 取其中三个特征做分析演示，分别是：
# pclass，1-一等舱，2-二等舱，3-三等舱
# age年龄
# sex性别
x = train_data[['length', 'topic_match', 'senti']]
x = x[:5000]
print("*" * 30 + " x " + "*" * 30)
print(x.info())
print(x.head())

# age字段存在缺失，用均值填充
# age_mean = x['Age'].mean()
# print("*" * 30 + " age_mean " + "*" * 30)
# print(age_mean)
# x['Age'].fillna(age_mean, inplace=True)
# print("*" * 30 + " 处理age缺失值后 " + "*" * 30)
# print(x.info())

# 特征抽取 - onehot编码
# 为了方便使用字典特征抽取，构造字典列表
x_dict_list = x.to_dict(orient='records')
print("*" * 30 + " train_dict " + "*" * 30)
print(pd.Series(x_dict_list[:5]))

dict_vec = DictVectorizer(sparse=False)
x = dict_vec.fit_transform(x_dict_list)
print("*" * 30 + " onehot编码 " + "*" * 30)
print(dict_vec.get_feature_names())
print(x[:5])

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 决策树分类器
dec_tree = DecisionTreeClassifier(criterion='gini',max_depth=10, min_impurity_decrease=0.002)
dec_tree.fit(x_train, y_train)
# dec_tree.predict()
tree_model = '.\\model\\tree_model'
dec_dump = pickle.dumps(dec_tree)
with open(tree_model,'wb') as model:
    model.write(dec_dump)

print('x_train')
print(x_train)
print('y_train')
print(y_train)

print("*" * 30 + " 准确率 " + "*" * 30)
print(dec_tree.score(x_test, y_test))
print(dec_tree.score(x_train, y_train))


def load_model():
    tree_model = '.\\model\\tree_model'
    with open(tree_model,'rb') as model:
        dec_dump = model.read()
    return dec_dump

def cv_score(d):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15)
    # clf = DecisionTreeClassifier(max_depth=d)
    dec_dump = load_model()
    clf = pickle.loads(dec_dump)
    # clf.fit(X_train, y_train)
    return(clf.score(X_train, y_train), clf.score(X_test, y_test))

#找到最合适的决策树层数
def test_best_depth():
    depths = np.arange(1,20)
    scores = [cv_score(d) for d in depths]
    tr_scores = [s[0] for s in scores]
    te_scores = [s[1] for s in scores]

    # 找出交叉验证数据集评分最高的索引
    tr_best_index = np.argmax(tr_scores)
    te_best_index = np.argmax(te_scores)

    print("bestdepth:", te_best_index+1, " bestdepth_score:", te_scores[te_best_index], '\n')
    depths = np.arange(1,20)
    plt.figure(figsize=(6,4), dpi=120)
    plt.grid()
    plt.xlabel('max depth of decison tree')
    plt.ylabel('Scores')
    plt.plot(depths, te_scores, label='test_scores')
    plt.plot(depths, tr_scores, label='train_scores')
    plt.legend()
    plt.show()
test_best_depth()
# # %matplotlib inline


# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# def minsplit_score(val):
#     clf = DecisionTreeClassifier(criterion='gini', min_impurity_decrease=val)
#     clf.fit(X_train, y_train)
#     return (clf.score(X_train, y_train), clf.score(X_test, y_test), )

# # 指定参数范围，分别训练模型并计算得分

# vals = np.linspace(0, 0.2, 100)
# scores = [minsplit_score(v) for v in vals]
# tr_scores = [s[0] for s in scores]
# te_scores = [s[1] for s in scores]

# bestmin_index = np.argmax(te_scores)
# bestscore = te_scores[bestmin_index]
# print("bestmin:", vals[bestmin_index])
# print("bestscore:", bestscore)

# plt.figure(figsize=(6,4), dpi=120)
# plt.grid()
# plt.xlabel("min_impurity_decrease")
# plt.ylabel("Scores")
# plt.plot(vals, te_scores, label='test_scores')
# plt.plot(vals, tr_scores, label='train_scores')

# plt.legend()
# plt.show()
# from sklearn.model_selection import GridSearchCV

# thresholds = np.linspace(0, 0.2, 50)
# param_grid = {'min_impurity_decrease':thresholds}

# clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
# clf.fit(x,y)

# print("best_parms:{0}\nbest_score:{1}".format(clf.best_params_, clf.best_score_))

# def plot_curve(train_sizes, cv_results, xlabel):
#     train_scores_mean = cv_results['mean_train_score']
#     train_scores_std = cv_results['std_train_score']
#     test_scores_mean = cv_results['mean_test_score']
#     test_scores_std = cv_results['std_test_score']
#     plt.figure(figsize=(6, 4), dpi=120)
#     plt.title('parameters turning')
#     plt.grid()
#     plt.xlabel(xlabel)
#     plt.ylabel('score')
#     plt.fill_between(train_sizes, 
#                      train_scores_mean - train_scores_std,
#                      train_scores_mean + train_scores_std, 
#                      alpha=0.1, color="r")
#     plt.fill_between(train_sizes, 
#                      test_scores_mean - test_scores_std,
#                      test_scores_mean + test_scores_std, 
#                      alpha=0.1, color="g")
#     plt.plot(train_sizes, train_scores_mean, '.--', color="r",
#              label="Training score")
#     plt.plot(train_sizes, test_scores_mean, '.-', color="g",
#              label="Cross-validation score")

#     plt.legend(loc="best")
#     plt.show()

# from sklearn.model_selection import GridSearchCV

# thresholds = np.linspace(0, 0.2, 50)
# # Set the parameters by cross-validation
# param_grid = {'min_impurity_decrease': thresholds}

# clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
# clf.fit(x, y)
# print("best param: {0}\nbest score: {1}".format(clf.best_params_, 
#                                                 clf.best_score_))

# # plot_curve(thresholds, clf.cv_results_, xlabel='gini thresholds')

# from sklearn.model_selection import GridSearchCV

# entropy_thresholds = np.linspace(0, 1, 100)
# gini_thresholds = np.linspace(0, 0.2, 100)
# #设置参数矩阵：
# param_grid = [{'criterion': ['entropy'], 'min_impurity_decrease': entropy_thresholds},
#               {'criterion': ['gini'], 'min_impurity_decrease': gini_thresholds},
#               {'max_depth': np.arange(2,10)},
#               {'min_samples_split': np.arange(2,30,2)}]
# clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
# clf.fit(x, y)
# print("best param:{0}\nbest score:{1}".format(clf.best_params_, clf.best_score_))


# from sklearn.datasets import load_iris
# from sklearn import tree
 
# iris = load_iris()
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)
 
 
# from six import StringIO
# import pydot
 
# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data) 
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph[0].write_dot('iris_simple.dot')
# graph[0].write_png('iris_simple.png')
