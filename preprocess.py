import os
import json
import glob
import unicodedata
from collections import deque
import lzma
import aiohttp
import asyncio
from tqdm import tqdm


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


async def lang_detect(dictToSend):
    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:5002/batch_lang_detect',
                                json=dictToSend) as resp:
            return await resp.json()


async def tokenize(dictToSend):
    async with aiohttp.ClientSession() as session:
        async with session.post('http://localhost:5002/batch_tokenize',
                                json=dictToSend) as resp:
            return await resp.json()


def batch_preprocess(docs):
    temp = deque()
    while len(docs) > 0:
        temp.append(docs.pop())
        if len(temp) > 25000:
            preprocess(list(temp))
            temp = deque()
    preprocess(list(temp))


def preprocess(docs):
    dictToSend = {'lang': lang, 'docs': docs}
    loop = asyncio.get_event_loop()
    res = loop.run_until_complete(lang_detect(dictToSend))
    detected_langs = res["detected_langs"]

    docs = [doc for doc, lang in zip(docs, detected_langs) if lang == 'en']
    if len(docs) > 0:
        dictToSend = {'lang': lang, 'docs': docs}
        res = loop.run_until_complete(tokenize(dictToSend))
        if "tokens" in res:
            docs = [" ".join(d).lower() for d in res["tokens"]]
            with open(os.path.join(interimpath, "documents_"+lang), "a") as outfile:
                outfile.write("\n".join(docs) + "\n")


lang = "en"
rawpath = "/home/chris/data/CORE/fulltext/"
interimpath = "/home/chris/data/CORE/interim/"
modelpath = "models"
rawfiles = glob.glob(rawpath+"*.json.xz")

with open("done.txt", "r") as infile:
    done = set(infile.read().splitlines())
rawfiles = [r for r in rawfiles if r not in done]

for r in tqdm(rawfiles, desc="collections", mininterval=10, maxinterval=120):
    with lzma.open(r) as infile:
        docs = [extract_text(json.loads(l.decode('utf-8'))) for l in infile]
    batch_preprocess(docs)
    with open("done.txt", "a") as outfile:
        outfile.write(r+"\n")
