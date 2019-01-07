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
        self.langdetect = spacy.load('en_core_web_sm')
        self.langdetect.add_pipe(language_detector)
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
            for l in infile:
                if len(l) > 0:
                    try:
                        raw = json.loads(l.decode('utf-8'))
                        text = self.extract_text(raw)
                        tag = self.extract_tag(raw)
                        doc = self.langdetect(text)
                        tokens = self.tokenize(doc)
                        if self.langfilter:
                            if doc._.languages[0] == self.langfilter:
                                yield TaggedDocument(tokens, [tag])
                        else:
                            yield TaggedDocument(tokens, [tag])
                    except Exception as e:
                        # print(e, tag)
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
