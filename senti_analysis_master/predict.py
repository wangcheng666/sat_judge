#!/usr/bin/env python
# -*- coding: utf-8  -*-
# PCA  SVM
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn import metrics
import pickle
import os

from senti_analysis_master.getwordvecs import genVecsOfSent

def get_trained_svm_model():
    file_path = os.path.dirname(__file__)

    svm_model = file_path + '\\model\\svm_model'
    with open(svm_model,'rb') as model:
        svm_dump = model.read()
        svm = pickle.loads(svm_dump)
    return svm


# print ('Test Accuracy: %.2f'% clf.score(x_pca,y))

#Create ROC curve
# pred_probas = clf.predict_proba(x_pca)[:,1] #score

# fpr,tpr,_ = metrics.roc_curve(y, pred_probas)
# roc_auc = metrics.auc(fpr,tpr)
# plt.plot(fpr, tpr, label = 'area = %.2f' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.legend(loc = 'lower right')
# plt.show()

def predict(file,clf):

    pre_df = pd.read_csv(file)

    pre_x = pre_df.iloc[:,1:]
    # print('pre_x')
    # print(pre_x)
    n_components = 1
    # pca = PCA(n_components=n_components)
    # pca.fit(pre_x)


    # pre_x_pca = PCA(n_components = 100).fit_transform(pre_x)
    # print(pre_x_pca)



    # pred_result = clf.predict(pre_x_pca)
    pred_result = [1.0]
    return pred_result

def train():
    # 获取数据 [1995 rows x 400 columns]
    fdir = ''
    df = pd.read_csv(fdir + '2000_data.csv')
    y = df.iloc[:,1]
    x = df.iloc[:,2:]
    print(x)


    # PCA降维
    ##计算全部贡献率
    n_components = 400
    pca = PCA(n_components=n_components)
    pca.fit(x)
    #print pca.explained_variance_ratio_

    ##PCA作图
    # plt.figure(1, figsize=(4, 3))
    # plt.clf()
    # plt.axes([.2, .2, .7, .7])
    # plt.plot(pca.explained_variance_, linewidth=2)
    # plt.axis('tight')
    # plt.xlabel('n_components')
    # plt.ylabel('explained_variance_')
    # plt.show()



    ##根据图形取100维
    x_pca = PCA(n_components = 100).fit_transform(x)


    # SVM (RBF)
    # using training data with 100 dimensions

    clf = svm.SVC(C = 2, probability = True)
    clf.fit(x_pca,y)
    svm_model = '.\\model\\svm_model'
    dec_dump = pickle.dumps(clf)
    with open(svm_model,'wb') as model:
        model.write(dec_dump)

    return clf


def predict_senti(sentence, model, svm):
    sentenceInput = genVecsOfSent(sentence,model)
    X = sentenceInput[:]
    # write in file   
    df_x = pd.DataFrame(X)
    # df_y = pd.DataFrame(Y)
    data = pd.concat([df_x],axis = 1)
    #print data
    data.to_csv('answer_data.csv')
    pre_result = predict('answer_data.csv',svm)

    return pre_result[0]



if __name__ == '__main__':
    clf = train()
    answer_results = predict('answer_data.csv',clf)
    comment_results= predict('comment_data.csv',clf)
    answer_result_file = 'answer_result.txt'
    comment_result_file = 'comment_result_file'
    with open(answer_result_file,'w',encoding='utf8') as result:
        for i in answer_results:
            result.write(str(i)+'\n')
    with open(comment_result_file,'w',encoding='utf8') as result:
        for i in comment_results:
            result.write(str(i)+'\n')