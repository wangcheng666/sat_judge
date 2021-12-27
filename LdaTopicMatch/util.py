#coding=utf-8
import csv
import codecs
import os,sys
import re
import importlib,sys
importlib.reload(sys)
from gensim.models import LdaModel,TfidfModel,LsiModel, ldamodel
from gensim import similarities
from gensim import corpora
import jieba
import jieba.analyse
import codecs,sys,string,re

from numpy import byte

def getdata(sourceFile,targetFile1,targetFile2,targetFile):
    data = csv.reader(open(sourceFile,'r'))
    target = codecs.open(targetFile, 'w', encoding='utf-8')
    target1 = codecs.open(targetFile1, 'w', encoding='utf-8')
    target2 = codecs.open(targetFile2, 'w', encoding='utf-8')

    length_file = 'length.txt'
    y_file = 'y.txt'
    length_f = codecs.open(length_file, 'w', encoding='utf-8')
    y_f = codecs.open(y_file, 'w', encoding='utf-8')

    for i,line in enumerate(data):
        if i != 0:
            comment = line[11].strip()
            answer = line[14].strip()
            length = line[15]
            y = line[24]
            
            if comment != '' and answer != '':
                target1.write(comment+'\n')
                target.write(comment+'\n')
                target2.write(answer+'\n')
                target.write(answer+'\n')
                length_f.write(length + '\n')
                y_f.write(y + '\n')
    target1.close()
    target2.close()

def create_data(corpus_path):#构建数据，先后使用doc2bow和tfidf model对文本进行向量表示
    sentences = []
    with open(corpus_path, 'r', encoding='utf8') as corpus:
        sentences = [[word for word in line.strip().lower().split()] for line in corpus]
    #对文本进行处理，得到文本集合中的词表
    dictionary = corpora.Dictionary(sentences)
    dictionary_file = '.\\Dic\\train_dic'
    dictionary.save(dictionary_file)
    
    #利用词表，对文本进行cbow表示
    corpus = [dictionary.doc2bow(text) for text in sentences]
    #利用cbow，对文本进行tfidf表示
    corpus_tfidf = TfidfModel(corpus)[corpus]
    return dictionary, corpus, corpus_tfidf


def trained_lda_model(dictionary,corpus_tfidf,cluster_keyword_lda):#使用lda模型，获取主题分布   
    lda = LdaModel(corpus=corpus_tfidf, num_topics=30, id2word=dictionary,
                            alpha=0.01, eta=0.01, minimum_probability=0.001,
                            update_every = 1, chunksize = 100, passes = 1)
    # lda = LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=11)
    f_keyword = open(cluster_keyword_lda, 'w+',encoding= 'utf8')
    for topic in lda.print_topics(30,53):
        words=[]
        for word in topic[1].split('+'):
            word=word.split('*')[1].replace(' ','')
            words.append(word)
        f_keyword.write(str(topic[0])+'\t'+','.join(words)+'\n')
    #利用lsi模型，对文本进行向量表示，这相当于与tfidf文档向量表示进行了降维，维度大小是设定的主题数目  
    
    # temp_file = 'temp.txt'
    # with open(temp_file,'w+',encoding='utf8') as temp:

    #     corpus_lda = lda[corpus_tfidf]
    #     for doc in corpus_lda:
    #         temp.write(str(len(doc))+'\t' +','.join(doc) + '\n')
    return lda


def get_trained_lda_model():
    """
    获得训练后的lda模型
    """
    file_path = os.path.dirname(__file__)
    model_file = file_path + '\\model\\lda_model'
    lda = LdaModel.load(model_file)
    return lda

def get_lda_dic():
    """
    获得训练数据生成的字典
    """
    file_path = os.path.dirname(__file__)
    print(file_path)
    dic_file = file_path + '\\Dic\\train_dic'
    dic = corpora.Dictionary.load(dic_file)
    return dic

# 文本分词
def prepareData(sourceFile,targetFile):
    f = codecs.open(sourceFile, 'r', encoding='utf-8')
    target = codecs.open(targetFile, 'w', encoding='utf-8')
    print ('open source file: '+ sourceFile)
    print ('open target file: '+ targetFile)

    lineNum = 1
    line = f.readline()
    while line:
        print ('---processing ',lineNum,' article---')
        line = clearTxt(line)
        seg_line = sent2word(line)
        target.writelines(seg_line + '\n')       
        lineNum = lineNum + 1
        line = f.readline()
    print ('well done.')
    f.close()
    target.close()

# 清洗文本
def clearTxt(line):
    if line != '': 
        line = line.strip()
        intab = b""
        outtab = b""
        trantab = bytes.maketrans(intab, outtab)
        pun_num =string.punctuation + string.digits
        pun_num = bytes(pun_num,encoding='utf8')
        line = line.encode('utf-8')
        line = line.translate(trantab,pun_num)
        line = line.decode("utf8")
        #去除文本中的英文和数字
        line = re.sub("[a-zA-Z0-9]","",line)
        #去除文本中的中文符号和英文符号
        line = re.sub("[\s+\.\!\/_,$%^*(+\"\'；：“”．]+|[+——！，。？?、~@#￥%……&*（）]+".encode("utf8").decode("utf8"), "",line) 
    return line

#文本切割
def sent2word(line):
    segList = jieba.cut(line,cut_all=False)    
    segSentence = ''
    for word in segList:
        if word != '\t':
            segSentence += word + " "
    return segSentence.strip()

def topic_match(comment_file, answer_file, result_file, dictionary, lda):
    with open(comment_file, 'r', encoding='utf8') as comments,\
    open(answer_file, 'r', encoding= 'utf8') as answers,\
    open(result_file, 'w+', encoding='utf8') as result:
        for comment,answer in zip(comments,answers):
            comment = clearTxt(comment)
            answer = clearTxt(answer)
            comment_cut = sent2word(comment)
            answer_cut = sent2word(answer)
            result.write(comment + '\n')
            result.write(answer + '\n')
            if match(comment_cut, answer_cut, dictionary, lda):
                result.write(str(1) + '\n')
            else:
                result.write(str(0) + '\n')
            

def match(comment, answer,dictionary, lda):
    sentences = []

    comment = comment.strip()
    sentences.append(comment.split(' '))
    answer = answer.strip()
    sentences.append(answer.split(' '))
    #利用词表，对文本进行cbow表示
    corpus = [dictionary.doc2bow(text) for text in sentences]
    comment_cor = corpus[0]
    answer_cor = corpus[1]
    vector_comment = lda[comment_cor]
    vector_answer = lda[answer_cor]
    vector_comment.sort(key = takeSecond,reverse = True)
    vector_answer.sort(key = takeSecond,reverse = True)
    count = 0
    main_topics_comment = []
    for i,topic in enumerate(vector_comment):
        # if i <= 15:
        main_topics_comment.append(topic[0])

    main_topics_answer = []
    for i,topic in enumerate(vector_answer):
        # if i <= 15:
        main_topics_answer.append(topic[0])

    # print(main_topics_comment)
    # print(main_topics_answer)
    for topic_answer in main_topics_answer:
        if topic_answer in main_topics_comment:
            count+=1
    # for topic_comment, topic_answer in zip(vector_comment, vector_answer):
    #     if topic_comment[0] == topic_answer[0]:
    #         count += 1
    if count > 2:
        return True
    return False


def takeSecond(elem):
    return elem[1]


# '''模型复杂度和主题一致性提供了一种方便的方法来判断给定主题模型的好坏程度。特别是主题一致性得分更有帮助。'''
# model_list = []
# perplexity = []
# coherence_values = []
# for num_topics in range(2,21,1):
#     lda_model = models.LdaModel(corpus=corpus, id2word=dictionary,  random_state=1, num_topics=num_topics,
#     random_state=100,update_every=1,chunksize=100,passes=10, alpha='auto', per_word_topics=True )   
#     model_list.append(lda_model)
#     #计算困惑度    
#     perplexity_values = lda_model.log_perplexity(corpus)    
#     print('%d 个主题的Perplexity为: ' % (num_topics-1), perplexity_values) 
#     # a measure of how good the model is. lower the better.    
#     perplexity.append(perplexity_values)
#     #计算一致性    
#     coherencemodel = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
#     coherence_values.append(round(coherencemodel.get_coherence(),3))    
#     print('%d 个主题的Coherence为: ' % (num_topics-1), round(coherencemodel.get_coherence(),3))

