#!/usr/bin/env python
# -*- coding: utf-8  -*-
#从词向量模型中提取文本特征向量
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import logging
import os.path
import codecs,sys
import numpy as np
import pandas as pd
import gensim

# 返回特征词向量
def getWordVecs(wordList,model):
    vecs = []
    for word in wordList:
        word = word.replace('\n','')
        #print word
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')
    

# 构建文档词向量 
def buildVecs(filename,model):
    fileVecs = []
    with codecs.open(filename, 'rb', encoding='utf-8') as contents:
        for line in contents:
            logger.info("Start line: " + line)
            wordList = line.split(' ')
            vecs = getWordVecs(wordList,model)
            #print vecs
            #sys.exit()
            # for each sentence, the mean vector of all its vectors is used to represent this sentence
            if len(vecs) >0:
                vecsArray = sum(np.array(vecs))/len(vecs) # mean
                #print vecsArray
                #sys.exit()
                fileVecs.append(vecsArray)
    return fileVecs   

def genVecsOfSent(sentence:str,model):
    fileVecs = []
    wordList = sentence.split(' ')
    vecs = getWordVecs(wordList,model)
    if len(vecs) >0:
        vecsArray = sum(np.array(vecs))/len(vecs)
        fileVecs.append(vecsArray)
    return fileVecs

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    
    # load word2vec model
    fdir = 'D:\\桌面应用\\学习\\大三上\\自然语言理解初步\\商家回复满意度自动判断\\senti_analysis-master\\'
    inp = fdir + 'wiki.zh.text.vector'
    model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)
    
    answerInput = buildVecs(fdir + 'answer_cut.txt',model)
    negInput = buildVecs(fdir + '2000_neg_cut_stopword.txt',model)

    # use 1 for positive sentiment， 0 for negative
    # Y = np.concatenate((np.ones(len(posInput)), np.zeros(len(negInput))))
    # Y = np.concatenate((np.ones(len(answerInput))))

    X = answerInput[:]
    print (X)
    # for neg in negInput:
    #     X.append(neg)
    # X = np.array(X)

    # write in file   
    df_x = pd.DataFrame(X)
    # df_y = pd.DataFrame(Y)
    data = pd.concat([df_x],axis = 1)
    #print data
    data.to_csv(fdir + 'answer_data.csv')
    

    


