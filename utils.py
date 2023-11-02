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
    return random.choice(list(synonyms)) if synonyms else word

def change_number(word, tag):
    if tag.startswith('NN'):
        lemmas = wordnet.synsets(word, pos='n')
        if not lemmas:
            return word  # If word has no synsets in WordNet, return as is
        
        lemma = lemmas[0]  # Take the first lemma
        if tag == 'NNS':
            # Use NLTK's morphy to find the base form
            singular_form = wordnet.morphy(word, wordnet.NOUN)
            return singular_form if singular_form else word
        elif tag == 'NN':
            # Unfortunately, NLTK does not provide a way to pluralize, so this might not be accurate for all nouns
            plural_form = lemma.name() + 's'
            return plural_form
    return word

def custom_transform(example):
    words = word_tokenize(example['text'])
    pos_tags = pos_tag(words)

    transformed_words = []
    for word, tag in pos_tags:
        if tag in ('NN', 'NNS'):
            # Attempt to change the number
            new_word = change_number(word, tag)
        else:
            # Attempt to get a synonym
            new_word = get_synonyms(word, tag)
        transformed_words.append(new_word)

    example['text'] = ' '.join(transformed_words)
    return example