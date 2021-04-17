import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet')
nltk.download('punkt')

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

wnl = nltk.WordNetLemmatizer()


def might_be_synonyms(w1, w2):
    s1 = set()
    d1 = set()
    s2 = set()
    d2 = set()
    for synset in wn.synsets(w1):
        lemma_names = synset.lemma_names()
        s1.update(lemma_names)
        d1.update(nltk.word_tokenize(synset.definition()))
    for synset in wn.synsets(w2):
        lemma_names = synset.lemma_names()
        s2.update(lemma_names)
        d2.update(nltk.word_tokenize(synset.definition()))
    total_intersection = len(s1.intersection(s2)) + len(d1.intersection(s2)) + len(d2.intersection(s1))
    return total_intersection > 0


def get_synonyms(word):
    synsets = wn.synsets(word)
    d = {'word_1': [], 'word_2': []}
    if synsets:
        synonyms = synsets[0].lemma_names()
        for synonym in synonyms:
            d['word_1'].append(word)
            d['word_2'].append(synonym)
    return pd.DataFrame(d)


class SiameseSynonymsNN(nn.Module):
    pass


if __name__ == "__main__":
    # print(might_be_synonyms('car', 'airplane'))
    synsets = wn.synsets('car')
    print(synsets[0].lemma_names())
    synsets = wn.synsets('auto')
    print(synsets)

    word_strings = np.loadtxt('glove.tokens.strings.txt', dtype=object)
    df = pd.DataFrame(word_strings, columns=['words'])
    print(df.shape)



