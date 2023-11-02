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

def get_wordnet_pos(treebank_tag):
    """
    Return the WordNet POS tag from the Penn Treebank tag.
    """
    ADJ, NOUN, ADV, VERB = 'a', 'n', 'r', 'v'
    if treebank_tag.startswith('J'):
        return ADJ
    elif treebank_tag.startswith('V'):
        return VERB
    elif treebank_tag.startswith('R'):
        return ADV
    elif treebank_tag.startswith('N'):
        return NOUN
    else:
        return ''

def get_synonyms(word, pos):
    """
    Get synonyms for the word with the given part of speech.
    """
    synonyms = set()
    for synset in wordnet.synsets(word, pos=pos):
        for lemma in synset.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name().replace('_', ' '))
    return random.choice(list(synonyms)) if synonyms else word

def custom_transform(example):
    words = word_tokenize(example['text'])
    pos_tags = pos_tag(words)

    transformed_words = []
    for word, tag in pos_tags:
        wordnet_pos = get_wordnet_pos(tag)
        if wordnet_pos:
            # Attempt to get a synonym
            new_word = get_synonyms(word, wordnet_pos)
        else:
            new_word = word
        transformed_words.append(new_word)

    example['text'] = ' '.join(transformed_words)
    return example