import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk import pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def get_hypernym(word):
    synsets = wordnet.synsets(word)
    if synsets:
        hypernyms = synsets[0].hypernyms()
        if hypernyms:
            return hypernyms[0].lemmas()[0].name()
    return word

def custom_transform(example):
    # Tokenize and tag the sentence
    words = word_tokenize(example['text'])
    pos_tags = pos_tag(words)

    # Replace nouns and verbs with their hypernyms
    transformed_words = []
    for word, tag in pos_tags:
        if tag.startswith('NN') or tag.startswith('VB'):
            hypernym = get_hypernym(word)
            transformed_words.append(hypernym)
        else:
            transformed_words.append(word)

    # Detokenize the transformed sentence
    example['text'] = TreebankWordDetokenizer().detokenize(transformed_words)
    return example