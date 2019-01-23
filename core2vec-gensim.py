import os
from gensim.models import Word2Vec
from nltk.corpus import stopwords


data_folder_name = '/home/chris/data/CORE/interim'

stops = stopwords.words('english')

# print("Loading data")
# with open(os.path.join(data_folder_name, "documents_en")) as infile:
#     docs = ([t for t in doc.lower().split() if t not in stops]
#             for doc in infile)

model = Word2Vec(size=300, window=8,  # w2v params
                 min_count=20,  # vocab params
                 sg=1, hs=0, negative=10,  # loss & sampling params
                 workers=10, iter=1)
print("Building vocabulary")
model.build_vocab(corpus_file=os.path.join(data_folder_name, "documents_en"))
print("Starting training")
with open(os.path.join(data_folder_name, "documents_en")) as infile:
    docs = ([t for t in doc.lower().split() if t not in stops]
            for doc in infile)
model.train(corpus_file=os.path.join(data_folder_name, "documents_en"),
            total_examples=model.corpus_count,
            total_words=len(model.wv.vocab),
            epochs=1)
model = model.wv
model.save("models/w2v_model_en")
