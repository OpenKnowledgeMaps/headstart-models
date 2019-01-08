import os
import glob
from corpora.core import CORECorpus
import random

lang = 'en'
rawpath = "/home/chris/data/CORE/fulltext/"
interimpath = "/home/chris/data/CORE/interim/"
modelpath = "models"
rawfiles = glob.glob(rawpath+"*.json.xz")
corpus = CORECorpus(random.sample(rawfiles, 200))
corpus.set_lang_filter(lang)
with open(os.path.join(interimpath, "documents_"+lang), "a") as outfile:
    for doc in corpus.tagged_doc_stream_from_corpus():
        outfile.write(doc.words+"\n")
