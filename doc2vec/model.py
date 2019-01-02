from gensim.models.doc2vec import Doc2Vec
from tqdm import tqdm


def build_model(corpus):
    model = Doc2Vec(documents=None,
                    vector_size=300, window=5,
                    min_count=50, workers=4)
    model.build_vocab(corpus.tagged_doc_stream_from_corpus())
    for c in tqdm(corpus.collections, desc="collections"):
        docs = list(corpus.tagged_doc_stream(c))
        total_words = sum([len(d.words) for d in docs])
        model.train(docs, total_words=total_words, epochs=1)
    model.delete_temporary_training_data(keep_doctags_vectors=True,
                                         keep_inference=True)
    return model