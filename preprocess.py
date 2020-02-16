# pylint: disable=E1101

import re
import os
import pickle
import pandas as pd
import preprocessor as p

import nltk
from nltk.tokenize import TweetTokenizer


# loading stop words from local
# stop words is pre download from nltk
with open(os.getcwd() + '/stopwords.json', 'rb') as f:
    stop_words = pickle.load(f)
tknzr = TweetTokenizer(reduce_len=True, strip_handles=True)


def clean_text(t):
    """
    remove redundant text from tweets
    only return alphbets, numbers, !, and ?

    Arguments:
        t {[str]} -- any tweets

    Returns:
        t [str] -- clean tweets
    """
    p.set_options(
        p.OPT.URL,
        p.OPT.MENTION,
        p.OPT.HASHTAG,
        p.OPT.RESERVED,
        p.OPT.EMOJI,
        p.OPT.SMILEY,
    )
    t = p.clean(t)
    t = re.sub(r"[\\//_,;.:*+\-\>\<)(%^$|~&`'\"\[\]\=]+", '', t)
    t = re.sub(r'[^\x00-\x7F]+', ' ', t)
    return t.lower()


def tokenize_text(t, tokenizer=tknzr, stop_words=stop_words):
    """
    tokenize preprocessed tweets

    Arguments:
        t {[str]} -- clean tweets

    Keyword Arguments:
        tokenizer {[nltk tokenizer]} -- one of nltk tokenizer (default: {TweetTokenizer})
        stop_words {[dict]} -- stop words map (default: {stop words})

    Returns:
        [list of str] -- ex. ['I', 'am', 'Richard', '!']
    """
    tList = [i for i in tokenizer.tokenize(t) if i not in stop_words]
    return tList


def load_embedding(path, max_length_dictionary=10000):
    """
    load embedding map 

    Arguments:
        path {[str]} -- the absolute path of where embedding map is

    Keyword Arguments:
        max_length_dictionary {int} -- maximum length of words to be loaded (default: {10000})

    Returns:
        [dict] -- embedding map loaded as python dictionary
    """
    embeddings_dict = {}
    i = 0

    with open(path, 'r') as f:
        for line in f:
            values = line.split()
            if values[0].isalnum():
                embeddings_dict[values[0]] = i
                i += 1

            if i == max_length_dictionary:
                break

    return embeddings_dict


def replace_token_with_index(tList, embeddingMap):
    """
    replace token with index in embedding map
    
    Arguments:
        tList {[list of str]} -- the output from tokenize_text
        embeddingMap {[dict]} -- the output from load_embedding or a self-defined dictionary
    
    Returns:
        [list of int] -- replace the tokens with index, ex. [1, 892, 3, 2467]
    """    
    tNewList = []
    for t in tList:
        # if t is not in EmbeddingMap continue the loop
        indice = embeddingMap.get(t)
        if not indice:
            continue
        else:
            tNewList.append(indice)
    return tNewList


def pad_sequence(tList, max_length_tweet=10):
    """
    construct pad sequence
    
    Arguments:
        tList {[list of int]} -- output from replace_token_with_index
    
    Keyword Arguments:
        max_length_tweet {int} -- the maximum length of tweet (default: {10})
    
    Returns:
        [list of int] -- return pad sequence list
    """    
    reLength = max_length_tweet - len(tList)
    
    if reLength > 0:
        tList.extend([0] * reLength)
    elif reLength < 0:
        tList = tList[:max_length_tweet]
        
    return tList