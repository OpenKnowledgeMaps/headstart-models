from corpora.corpus import FileCorpus
import string
import re


regex = re.compile('[%s]' % re.escape(string.punctuation))


class CORECorpus(FileCorpus):

    def get_tokens(self, raw):
        doc = []
        if raw.get("title"):
            doc.append(raw.get("title"))
        if raw.get("abstract"):
            doc.append(raw.get("abstract"))
        doc = " ".join(doc)
        doc = regex.sub('', doc)
        tokens = re.split(' |\r|\n', doc)
        return [t for t in tokens if len(t) > 0]
