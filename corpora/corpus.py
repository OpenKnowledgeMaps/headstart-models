import lzma
import json
from tqdm import tqdm
import spacy
from spacy_cld import LanguageDetector
from gensim.models.doc2vec import TaggedDocument

language_detector = LanguageDetector()


class FileCorpus(object):
    def __init__(self, files):
        self.collections = files
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe(language_detector)
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

    def tagged_doc_stream(self, f):
        with lzma.open(f) as infile:
            raw = [json.loads(l.decode('utf-8')) for l in infile]
            texts = [self.extract_text(r) for r in raw]
            tags = [self.extract_tag(r) for r in raw]
        docs = self.nlp.pipe(texts)
        for doc, tag in zip(docs, tags):
            try:
                if self.langfilter:
                    if doc._.languages[0] == self.langfilter:
                        yield TaggedDocument(self.tokenize(doc), [tag])
                else:
                    yield TaggedDocument(self.tokenize(doc), [tag])
            except Exception as e:
                print(e, tag)
                pass

    def tagged_doc_stream_from_corpus(self):
        for c in tqdm(self.collections, desc="collections from corpus"):
            docs = self.tagged_doc_stream(c)
            for d in docs:
                yield d


class DBCorpus(object):
    def __init__(self, db, collection):
        self.db = db
        self.collection = collection
