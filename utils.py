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

def get_synonym(word):
    synsets = wordnet.synsets(word)
    synonyms = set()
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.add(lemma.name().replace('_', ' '))
    synonyms.discard(word)
    return random.choice(list(synonyms)) if synonyms else word

def change_number(word, tag):
    if tag.startswith('NN'):
        if wordnet.synsets(word):
            lemmas = wordnet.synsets(word)[0].lemmas()
            if lemmas[0].count():
                if tag == 'NNS':
                    # Convert from plural to singular
                    return lemmas[0].name().replace('_', ' ')
                else:
                    # Convert from singular to plural
                    plural_form = lemmas[0].count_forms()[0]
                    return plural_form.replace('_', ' ')
    return word

def custom_transform(example):
    words = word_tokenize(example['text'])
    pos_tags = pos_tag(words)

    transformed_words = []
    for word, tag in zip(words, pos_tags):
        if tag.startswith('NN'):
            word = change_number(word, tag)
        elif tag.startswith(('JJ', 'RB', 'VB')):
            word = get_synonym(word)
        transformed_words.append(word)

    example['text'] = ' '.join(transformed_words)
    return example
