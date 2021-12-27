#coding=utf-8
import pickle
import LdaTopicMatch.util as Lda_tools
import gensim
import senti_analysis_master.predict as senp
from sklearn.feature_extraction import DictVectorizer
import os
import sys




def load_model():
    file_path = os.path.dirname(__file__)
    tree_model = file_path + '\\model\\tree_model'
    with open(tree_model,'rb') as model:
        dec_dump = model.read()
        dec_tree = pickle.loads(dec_dump)
    return dec_tree

def load_dic_model():
    file_path = os.path.dirname(__file__)
    dic_model = file_path + '\\model\\dic_model'
    with open(dic_model,'rb') as model:
        dic_dump = model.read()
        dic_tree = pickle.loads(dic_dump)
    return dic_tree

def format_data(comment:str, answer:str):
    """
    将评论和回复经过各种指标的量化组成决策树可以预测的格式
    """
    dic = Lda_tools.get_lda_dic()
    lda = Lda_tools.get_trained_lda_model()
    #文本清洗
    comment = Lda_tools.clearTxt(comment)
    answer = Lda_tools.clearTxt(answer)
    #文本切分
    comment_cut = Lda_tools.sent2word(comment)
    answer_cut = Lda_tools.sent2word(answer)
    #主题匹配
    topic_match = Lda_tools.match(comment=comment_cut, answer=answer_cut, dictionary=dic, lda=lda)
    #长度
    length = len(answer)
    #情感极性
    svm = senp.get_trained_svm_model()
    senti = senp.predict_senti(answer_cut,model,svm)
    
    pre_data = {}
    pre_data['topic_match'] = topic_match
    pre_data['senti'] = senti
    pre_data['length'] = length
    # feature_name = ['topic_match','senti','length']
    dv_train = DictVectorizer(sparse=False)
    x_data = dv_train.fit_transform(pre_data)
    
    return x_data

if __name__ == '__main__':
    print("模型加载中请稍等。。。")
    # file_path = os.path.dirname(__file__)
    # inp = file_path + '\\senti_analysis_master\\wiki.zh.text.vector'
    # model = gensim.models.KeyedVectors.load_word2vec_format(inp, binary=False)
    # dic_model = '.\\model\\dic_model'
    # dic_dump = pickle.dumps(model)
    # with open(dic_model,'wb') as model:
    #     model.write(dic_dump)
    model = load_dic_model()
    print("模型加载成功")
    comment = sys.argv[1]
    answer = sys.argv[2]
    # print (comment)
    # print(answer)
    x_data = format_data(comment , answer)

    dec_tree = load_model()

    result = dec_tree.predict(x_data)[0]
    print('result:(1代表满意，0代表不满意）' + result)

