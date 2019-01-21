import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords


data_folder_name = '/home/chris/data/CORE/interim'
window_size = 5

stops = stopwords.words('english')

print("Loading data")
with open(os.path.join(data_folder_name, "documents_en")) as infile:
    docs = ([t for t in doc.lower().split() if t not in stops]
            for doc in infile)
    tagged_docs = [TaggedDocument(doc, [tag])
                   for tag, doc in enumerate(docs)
                   if len(doc) > window_size]
total_examples = len(tagged_docs)

model = Doc2Vec(vector_size=300, window=window_size,
                min_count=50, max_vocab_size=100000,
                dbow_words=0,
                dm=1, workers=10, hs=0, negative=10)
print("Building vocabulary")
model.build_vocab(tagged_docs)
print("Starting training")
model.train(tagged_docs, total_examples=total_examples, epochs=1)
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
model.save("models/core2vec_model_en")
