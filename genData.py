#coding=utf-8
import LdaTopicMatch.util as Lda_tools
import csv
import pandas as pd

def genData():
    """
    获得决策树的训练数据
    """
    topic_match_file = '.\\LdaTopicMatch\\result.txt'
    senti_file = '.\\senti_analysis-master\\answer_result.txt'
    length_file = '.\\data\length.txt'
    sat_file = '.\\data\sat.txt'
    topic_matchs = []
    sentis = []
    lengths= []
    sats = []
    # data = csv.reader(open(,'r'))
    # for i,line in enumerate(data):
    #     if i != 0:
    #         answer_length = line[15]
    #         if answer_length != '':
    #             lengths.append(answer_length)

    with open(length_file, 'r', encoding='utf8') as length:
        for i,l in enumerate(length):
            lengths.append(l.strip())

    with open(topic_match_file,'r',encoding='utf8') as topic_match:
        for i,t in enumerate(topic_match):
            if i % 3 == 2:
                topic_matchs.append(t.strip())
    
    with open(senti_file,'r',encoding='utf8') as senti:
        for i,s in enumerate(senti):
            sentis.append(s.strip())

    with open(sat_file,'r',encoding='utf8') as sat:
        for i,s in enumerate(sat):
            sats.append(s.strip())

    print(len(sentis))
    print(len(topic_matchs))
    print(len(lengths))

    Data = {}
    Data['topic_match'] = topic_matchs
    Data['senti'] = sentis
    Data['length'] = lengths
    Data['sat'] = sats
    Data_fr = pd.DataFrame(Data)
    print(Data_fr)
    Data_fr.to_csv('.\\data\\dt_train_data.csv')
    

def makeCase(comment: str, answer: str) -> list:
    """
    通过一对评论和回复生成预测用例
    """
    #获得商家恢复和用户评论的主题匹配度
    model_file = '.\\LdaTopicMatch\\model\\lda_model'
    lda = Lda_tools.get_trained_lda_model(model_file)
    dic_file = 'L.\\daTopicMatch\\Dic\\train_dic'
    dic = Lda_tools.get_lda_dic(dic_file)
    topic_match = Lda_tools.match(comment=comment, answer=answer, dictionary=dic, lda=lda)

    #获得商家恢复的长度

    #获得商家恢复的情感极性

if __name__ == '__main__':
    genData()



