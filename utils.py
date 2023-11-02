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

def find_adverbial_phrases(tags):
    # This function identifies adverbial phrases.
    adverbial_phrases = []
    for i, (word, tag) in enumerate(tags):
        if tag.startswith('RB'):  # RB, RBR, and RBS are adverb related POS tags
            adverbial_phrases.append((i, word))
    return adverbial_phrases

def rearrange_adverbial_phrases(sentence):
    tokens = word_tokenize(sentence)
    tags = pos_tag(tokens)
    
    adverbial_phrases = find_adverbial_phrases(tags)
    
    # Move the first adverb to the start of the sentence if it's not already there
    if adverbial_phrases and adverbial_phrases[0][0] > 0:
        adverb, adverb_idx = adverbial_phrases[0]
        # Remove the adverb
        tokens.pop(adverb_idx)
        # Insert the adverb at the beginning
        tokens.insert(0, adverb)
    
    return ' '.join(tokens)

def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINS HERE ####

    # Get the original sentence from the example
    sentence = example["text"]
    # Transform the sentence by rearranging adverbial phrases
    transformed_sentence = rearrange_adverbial_phrases(sentence)
    # Update the example text with the transformed sentence
    example["text"] = transformed_sentence

    ##### YOUR CODE ENDS HERE ######
    return example
