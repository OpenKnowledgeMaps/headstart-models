import lzma
import json
from tqdm import tqdm
import spacy
from pycld2 import detect, error as pycld_error
from spacy_cld import LanguageDetector
from gensim.models.doc2vec import TaggedDocument

language_detector = LanguageDetector()


class FileCorpus(object):
    def __init__(self, files):
        self.collections = files
        self.langdetect = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
        self.langdetect.add_pipe(language_detector)
        self.nlp = spacy.load('en_core_web_sm')
        self.langfilter = None

    def set_lang_filter(self, lang):
        self.langfilter = lang

    @staticmethod
    def extract_text(raw):
        raise NotImplementedError

    @staticmethod
    def extract_tag(raw):
        raise NotImplementedError

    def tokenize(self, text):
        raise NotImplementedError

    def docs(self, f):
        with lzma.open(f) as infile:
            raw = [json.loads(l.decode('utf-8')) for l in infile]
        texts = [self.extract_text(r) for r in raw]
        docs = self.langdetect.pipe(texts, batch_size=1000, n_threads=5)
        docs = [doc.text for doc in docs
                if (len(doc._.languages) > 0
                    and doc._.languages[0] == self.langfilter)
                ]
        docs = self.nlp.pipe(docs, batch_size=1000, n_threads=5)
        return "\n".join([doc.text for doc in docs])

    def docs_from_collections(self):
        for c in tqdm(self.collections, desc="collections from corpus"):
            yield self.docs(c)


class DBCorpus(object):
    def __init__(self, db, collection):
        self.db = db
        self.collection = collection
