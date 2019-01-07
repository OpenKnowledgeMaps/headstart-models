import string
import re
from corpora.corpus import FileCorpus


regex = re.compile('[%s]' % re.escape(string.punctuation))


class CORECorpus(FileCorpus):

    @staticmethod
    def extract_text(raw):
        text = []
        if raw.get("title"):
            text.append(raw.get("title"))
        if raw.get("abstract"):
            text.append(raw.get("abstract"))
        text = ". ".join(text)
        return text

    @staticmethod
    def extract_tag(raw):
        return raw.get("coreId")

    def tokenize(self, doc):
        return " ".join([t.text for t in doc if t.is_alpha])
