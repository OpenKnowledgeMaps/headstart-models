import lzma
import json
from tqdm import tqdm
from gensim.models.doc2vec import TaggedDocument


class FileCorpus(object):
    def __init__(self, files):
        self.collections = files

    def get_tokens(self, raw):
        raise NotImplementedError

    def get_tagged_doc(self, raw):
        raw = json.loads(raw)
        tokens = self.get_tokens(raw)
        tag = raw.get("coreId")
        return TaggedDocument(tokens, [tag])

    def tagged_doc_stream(self, f):
        with lzma.open(f) as infile:
            for l in infile:
                if len(l) > 0:
                    doc = self.get_tagged_doc(l.decode('utf-8'))
                    if len(doc.words) > 0:
                        yield doc

    def tagged_doc_stream_from_corpus(self):
        for c in tqdm(self.collections, desc="collections"):
            docs = self.tagged_doc_stream(c)
            for d in docs:
                yield d


class DBCorpus(object):
    def __init__(self, db, collection):
        self.db = db
        self.collection = collection
