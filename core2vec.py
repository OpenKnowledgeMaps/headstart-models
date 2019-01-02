import glob
from corpora.core import CORECorpus
from doc2vec.model import build_model

filepath = "/home/chris/data/CORE/metadata/"
files = glob.glob(filepath+"*.json.xz")
corpus = CORECorpus(files)
model = build_model(corpus)
model.save("models/core2vec_model")
