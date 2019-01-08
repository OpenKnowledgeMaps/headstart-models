from gensim.models.doc2vec import Doc2Vec


def build_model(corpusfile):
    model = Doc2Vec(corpus_file=corpusfile,
                    vector_size=300, window=5,
                    min_count=50, workers=4)
    model.delete_temporary_training_data(keep_doctags_vectors=True,
                                         keep_inference=True)
    return model
