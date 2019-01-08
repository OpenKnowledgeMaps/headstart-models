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

    def tagged_docs(self, f):
        with lzma.open(f) as infile:
            raw = [json.loads(l.decode('utf-8')) for l in infile]
            texts = [self.extract_text(r) for r in raw]
            tags = [self.extract_tag(r) for r in raw]
        docs = self.langdetect.pipe(texts, batch_size=1000, n_threads=5)
        dt = [(doc, tag) for doc, tag in zip(docs, tags)
              if (len(doc._.languages) > 0
                  and doc._.languages[0] == self.langfilter)]
        texts = [d.text for d, t in dt]
        tags = [t for d, t in dt]
        docs = self.nlp.pipe(texts, batch_size=1000, n_threads=5)
        for doc, tag in zip(docs, tags):
            if doc:
                yield TaggedDocument(self.tokenize(doc), [tag])

    def tagged_doc_stream_from_corpus(self):
        for c in tqdm(self.collections, desc="collections from corpus"):
            docs = self.tagged_docs(c)
            for d in docs:
                yield d


class DBCorpus(object):
    def __init__(self, db, collection):
        self.db = db
        self.collection = collection
