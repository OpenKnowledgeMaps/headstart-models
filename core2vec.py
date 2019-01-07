import os
import glob
from corpora.core import CORECorpus
from doc2vec.model import build_model

lang = 'en'
rawpath = "/home/chris/data/CORE/fulltext/"
interimpath = "/home/chris/data/CORE/interim/"
modelpath = "models"
rawfiles = glob.glob(rawpath+"*.json.xz")
corpus = CORECorpus(rawfiles[:200])
corpus.set_lang_filter(lang)
model = build_model(corpus)
model.save(os.path.join(modelpath, "core2vec_model_"+lang+"_"))
