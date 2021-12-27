from gensim.models import LdaModel
from util import topic_match
from gensim import corpora


model_file = '.\\model\\lda_model'
lda = LdaModel.load(model_file)

corpus_file = '.\\data\\corpus_cut.txt'
sentences = []
for line in open(corpus_file, 'r', encoding='utf8'):
    line = line.strip()
    sentences.append(line.split(' '))

    #对文本进行处理，得到文本集合中的词表
dictionary = corpora.Dictionary(sentences)

comment_file = '.\\data\\comment.txt'
answer_file = '.\\data\\answer.txt'
result_file = 'result.txt'
topic_match(comment_file, answer_file, result_file, dictionary, lda)