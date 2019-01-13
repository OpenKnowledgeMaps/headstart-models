import unicodedata
from corpora.corpus import FileCorpus


class CORECorpus(FileCorpus):

    @staticmethod
    def extract_text(raw):
        text = []
        if raw.get("title"):
            text.append(raw.get("title"))
        if raw.get("abstract"):
            text.append(raw.get("abstract"))
        text = ". ".join(text)
        text = ''.join([l for l in text
                        if unicodedata.category(str(l))[0]
                        not in ('S', 'M', 'C')])
        return text

    @staticmethod
    def extract_tag(raw):
        return raw.get("coreId")

    def tokenize(self, doc):
        return " ".join([t.text.lower() for t in doc if t.is_alpha])
