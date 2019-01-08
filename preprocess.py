import os
import glob
from corpora.core import CORECorpus
import random

lang = 'en'
rawpath = "/home/chris/data/CORE/fulltext/"
interimpath = "/home/chris/data/CORE/interim/"
modelpath = "models"
rawfiles = glob.glob(rawpath+"*.json.xz")
corpus = CORECorpus(rawfiles)
corpus.set_lang_filter(lang)
with open(os.path.join(interimpath, "documents_"+lang), "a") as outfile:
    for docs in corpus.docs_from_collections():
        outfile.write(docs+"\n")
