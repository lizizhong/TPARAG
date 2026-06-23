# encoding = utf-8
import os
import  csv
import json
import re
import random
import string
import unicodedata
import nltk
nltk.download('wordnet')
from tqdm import tqdm
from glob import glob

from flair.data import Sentence
from flair.nn import Classifier

from nltk.corpus import wordnet as wordnet
from nltk import word_tokenize, sent_tokenize
import numpy as np

tagger_ner = Classifier.load('ner')
tagger_pos = Classifier.load('pos')


def read_instances(filename):
    '''read data from json file
    '''
    instances = {}
    with open(filename, 'r') as j:
        for line in j:
            instances.update(json.loads(line))

    return instances


def check_json_format(raw_msg):
    """
    determin whether a string confirms to the JSON format
    """
    if isinstance(raw_msg, str):  
        try:
            json.loads(raw_msg)
        except ValueError:
            print('Json False: ', raw_msg)
            return False
        print('True')
        return True
    else:
        print('String False')
        return False


def clean_json_format(msg):
    '''return a clean json format version for the given string
    '''
    try:
        begin_index, end_index = msg.index('{'), msg.rindex('}')
    except ValueError:
        return False

    return msg[begin_index: end_index+1]


def extract_corpus_texts(text):
    pattern = r"<corpus\d+> (.*?)(?=;\s*<corpus\d+>|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def read_retrieved_file(filename):
    instances = []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as j:
        for line in j:
            instances.append(json.loads(line))

    return instances


def write_malicious_passage(instance_list, filename):
    with open(filename, "w") as fw:
        if isinstance(instance_list, list):
            for item in instance_list:
                json.dump(item, fw)
                fw.write('\n')
        elif isinstance(instance_list, dict):
            json.dump(instance_list, fw)
            fw.write('\n')
    fw.close()


def is_punctuation(char):
    return char in string.punctuation


def isFloatNum(word_text):
    s = word_text.split('.')
    if len(s) > 2 or len(s) == 1:
        return False
    else:
        for si in s:
            if not si.isdigit():
                return False
        return True


def substution_vb_token(word_text):
    '''first find antonyms
    If there is no antonyms, find entailments
    If no antonyms and entailments, use a random lemma
    '''
    synset_words = wordnet.synsets(word_text)
    antonyms_words, entailment_words, lemma_words = [], [], []
    for synset_word in synset_words:
        synset_lemma_words = synset_word.lemmas()
        lemma_words.extend(synset_lemma_words)
    for lemma_word in lemma_words:
        antonyms_words.extend(lemma_word.antonyms())
        entailment_words.extend(lemma_word.entailments())
    if len(antonyms_words) > 0:
        randint = random.randint(0, len(antonyms_words)-1)
        substution_word = antonyms_words[randint]
        return substution_word._synset._lemma_names[0]
    elif len(entailment_words) > 0:
        substution_word = random.choice(entailment_words)
    elif len(lemma_words) > 0:
        filter_lemma = []
        for lemma in lemma_words:
            if 'verb' in lemma._synset._lexname and lemma._name not in word_text:
                filter_lemma.append(lemma._name)
        if len(filter_lemma) > 0:
            substution_word = random.choice(filter_lemma)
            return substution_word
        else:
            print('can not conduct vb substitutation:', word_text)
            return word_text
    else:
        print('can not conduct vb substitutation:', word_text)
        return word_text

    return substution_word._synset._lemma_names[0]


def substution_nn_token(word_text):
    '''Find the hypernums first
    Then random choose a hyponyms word
    If there is no hyponyms word, choose a hypernums word
    '''
    synset_words = wordnet.synsets(word_text)
    hypernyms_words = []
    for synset_word in synset_words:
        synset_hypernyms_words = synset_word.hypernyms()
        hypernyms_words.extend(synset_hypernyms_words)
    # hypernyms_words = synset_words.hypernyms()
    hyponyms_words = []
    for hypernyms_word in hypernyms_words:
        hyponyms_words.extend(hypernyms_word.hyponyms())
    if len(hyponyms_words) > 0:
        substution_word = random.choice(hyponyms_words)
    elif len(hypernyms_words) > 0:
        substution_word = random.choice(hypernyms_words)
    else:
        word_text_list = list(word_text)
        random.shuffle(word_text_list)
        shuffled_substution_word = ''.join(word_text_list)
        print('can not conduct nn substitutation:', word_text, shuffled_substution_word)
        return shuffled_substution_word
    
    return substution_word._lemma_names[0]


def substution_jj_or_rb_token(word_text):
    '''first find antonyms
    If there is no antonyms, find entailments
    If no antonyms and entailments, use a random lemma
    '''
    synset_words = wordnet.synsets(word_text)
    antonyms_words, entailment_words, lemma_words = [], [], []
    for synset_word in synset_words:
        synset_lemma_words = synset_word.lemmas()
        lemma_words.extend(synset_lemma_words)
    for lemma_word in lemma_words:
        antonyms_words.extend(lemma_word.antonyms())
        entailment_words.extend(lemma_word.entailments())
    if len(antonyms_words) > 0:
        randint = random.randint(0, len(antonyms_words)-1)
        substution_word = antonyms_words[randint]
    elif len(entailment_words) > 0:
        randint = random.randint(0, len(entailment_words)-1)
        substution_word = entailment_words[randint]
    elif len(lemma_words) > 0:
        filter_lemma = []
        for lemma in lemma_words:
            # TODO: filter word rules
            # if 'adj' in lemma._synset._lexname and lemma._name not in word_text:
            # if lemma._name not in word_text:
            filter_lemma.append(lemma._name)
        randint = random.randint(0, len(filter_lemma)-1)
        substution_word = filter_lemma[randint]
        return substution_word
    else:
        return word_text

    return substution_word._synset._lemma_names[0]


def substution_dt_cc_in_md_wdt_token(word_text, substution_set):
    substution_word = random.choice(substution_set)
    # while word_text == substution_set[randint]:
        # randint = random.randint(0, len(substution_set)-1)
    # substution_word = substution_set[randint]

    return substution_word


def substution_cd_token(word_text):
    '''random generate another int/float value
    '''
    random_num_words = ['one', 'two', 'three', 'four']
    if isFloatNum(word_text=word_text):
        float_word_prefix = round(float(word_text))
        random_num = random.randint(float_word_prefix-10, float_word_prefix+10) + \
                     random.random()
        return str(random_num)
    elif word_text.isalpha():
        rand_index = random.randint(0, len(random_num_words)-1)
        return random_num_words[rand_index]
    else:
        int_word_text = []
        for char in word_text:
            if char.isalpha() or is_punctuation(char):
                int_word_text.append(char)
            elif char.isdigit():
                try:
                    int_word = int(char)
                    random_num = random.randint(int_word-10, int_word+10)
                except Exception as e:
                    int_word = random.randint(-1000, 1000)
                    print(char)
            else:
                print('can not conduct cd substitutation:', word_text)
                return word_text
        return ''.join(int_word_text)
    
    
def substution_token_process(word, dt_word_set=None):
    if 'VB' in word.data_point.tag:
        substution_word = substution_vb_token(word_text=word.data_point.text)
    elif 'NN' in word.data_point.tag:
        substution_word = substution_nn_token(word_text=word.data_point.text)
    elif 'JJ' in word.data_point.tag or 'RB' in word.data_point.tag:
        substution_word = substution_jj_or_rb_token(word_text=word.data_point.text)
    elif 'CD' in word.data_point.tag:
        substution_word = substution_cd_token(word_text=word.data_point.text)
    elif 'DT' in word.data_point.tag:
        substution_word = substution_dt_cc_in_md_wdt_token(word_text=word.data_point.text, 
                                                           substution_set=dt_word_set)
    else:
        print(word.data_point.tag)
        substution_word = word.data_point.text

    return substution_word


def select_local_corpus(filename):
    corpus_array = []
    with open(filename, 'r') as reader:
        content = reader.read()
    content = content.split('\n')
    for line in content:
        line = line.strip()
        if len(line) > 0:
            corpus_array.append(line.strip())
    
    return corpus_array



def random_select_corpus(ner_type, from_file=True, instance_corpus=None):
    corpus_array = []
    ner_type = ner_type.lower()
    # read_file_dir = '/storage/zizhong/nanogcg/nanoGCG/data_process/wiki_ner_1027/per/wikiner_per_*'
    if from_file:
        read_file_dir = '/storage/zizhong/nanogcg/wiki_ner/wiki_ner_1027/' + ner_type + \
                        '/wikiner_' + ner_type + '_*'
        filenames = glob(read_file_dir)
        random_index = random.randint(0, len(filenames)-1)
        with open(filenames[random_index], 'r') as reader:
            content = reader.read()
        content = content.split('\n')
        for line in content:
            line = line.strip()
            if len(line) > 0:
                corpus_array.append(line.strip())
    else:
        per_corpus, loc_corpus, org_corpus, misc_corpus = [], [], [], []
        for ner_corpus in instance_corpus:
            for ner_entity in ner_corpus:
                tag, text = ner_entity.data_point.tag, ner_entity.data_point.text
                if tag == 'PER':
                    per_corpus.append(text)
                elif tag == 'LOC':
                    loc_corpus.append(text)
                elif tag == 'ORG':
                    org_corpus.append(text)
                else:
                    misc_corpus.append(text)
        corpus_array = [per_corpus, loc_corpus, org_corpus, misc_corpus]

    return corpus_array


def substution_ner_token(ner_type, per_corpus, loc_corpus, org_corpus, misc_corpus):
    assert ner_type in ['PER', 'LOC', 'ORG', 'MISC']
    if ner_type == 'PER':
        ner_corpus = per_corpus
    elif ner_type == 'LOC':
        ner_corpus = loc_corpus
    elif ner_type == 'ORG':
        ner_corpus = org_corpus
    else:
        ner_corpus = misc_corpus
    random_index = random.randint(0, len(ner_corpus)-1)
    substution_word = ner_corpus[random_index]
    # substution_word = list(substution_word)
    # random.shuffle(substution_word)
    # shuffled_substution_word = ''.join(substution_word)

    # return shuffled_substution_word
    return substution_word


def extract_span_numbers(text):
    # define the expression
    pattern = r'Span\[(\d+):(\d+)\]'
    match = re.search(pattern, text)
    if match:
        # extract and transfer to int
        start = int(match.group(1))
        end = int(match.group(2))
        return start, end
    else:
        return None

