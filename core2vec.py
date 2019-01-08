import os
import glob
from corpora.core import CORECorpus
from doc2vec.model import build_model
import random

lang = 'en'
rawpath = "/home/chris/data/CORE/fulltext/"
interimpath = "/home/chris/data/CORE/interim/"
modelpath = "models"
corpusfile = os.path.join(interimpath, "documents_"+lang)
model = build_model(corpusfile)
model.save(os.path.join(modelpath, "core2vec_model_"+lang))
