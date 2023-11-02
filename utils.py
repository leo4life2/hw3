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

def get_synonyms(word, tag):
    # Ensures the word is a noun, verb, adjective, or adverb for synonym replacement
    tag_map = {'N': 'n', 'J': 'a', 'V': 'v', 'R': 'r'}
    wn_tag = tag_map.get(tag[0].upper(), None)
    if wn_tag is None:
        return word

    synsets = wordnet.synsets(word, pos=wn_tag)
    if not synsets:
        return word

    synonyms = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))

    synonyms.discard(word)  # Remove the original word from synonyms
    return choice(list(synonyms)) if synonyms else word

def change_number(word, tag):
    if tag.startswith('N'):
        lemmas = wordnet.synsets(word)
        if not lemmas:
            return word
        lemma = lemmas[0]
        if tag == 'NNS':
            # If the word is plural, we will attempt to convert to singular
            singular_form = lemma.name().replace('_', ' ')
            return singular_form
        elif tag == 'NN':
            # If the word is singular, we will attempt to convert to plural
            # This simplistic approach may not work for all nouns
            plural_form = lemma.name().replace('_', ' ') + 's'
            return plural_form
    return word

def custom_transform(example):
    words = word_tokenize(example['text'])
    pos_tags = pos_tag(words)

    transformed_words = []
    for word, tag in pos_tags:
        if tag.startswith('NN'):
            # Attempt to change the number
            new_word = change_number(word, tag)
        else:
            # Attempt to get a synonym
            new_word = get_synonyms(word, tag)
        transformed_words.append(new_word)

    example['text'] = ' '.join(transformed_words)
    return example