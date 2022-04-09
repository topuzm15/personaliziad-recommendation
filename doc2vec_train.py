import pandas as pd
import multiprocessing
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


# doc2vec
corpus = pd.read_csv('data/corpus.csv', index_col=False)["doc"].apply(lambda x: x.replace('.','')).to_list()
Documents = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(corpus)]
epochs = 50
cores = multiprocessing.cpu_count()

model= Doc2Vec(dm=0,
               vector_size=256,
               negative=5,
               hs=0,
               min_count=2,
               sample = 0,
               workers=cores)

model.build_vocab(Documents)
for epoch in range(epochs):
    print('iteration {0}'.format(epoch),end = "\r")
    model.train(Documents,
                total_examples=model.corpus_count,
                epochs=1)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("model_data/d2v.model")
print("Model Saved.")