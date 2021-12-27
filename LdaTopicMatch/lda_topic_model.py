from util import trained_lda_model, create_data
 
 
corpus_path = '.\\data\\corpus_cut.txt'
dictionary, corpus, corpus_tfidf=create_data(corpus_path)
cluster_keyword_lda = '.\\data\\lda.txt'
lda = trained_lda_model(dictionary, corpus_tfidf, cluster_keyword_lda)

model_file = '.\\model\\lda_model'
lda.save(model_file)
