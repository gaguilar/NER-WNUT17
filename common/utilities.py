import re
import string
import csv
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from collections import defaultdict as ddict
from embeddings.twitter.word2vecReader import Word2Vec
from itertools import groupby
from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from settings import *

##############################################################
# General Functions
##############################################################

def unzip(list_of_tuples):
    return [list(elem) for elem in zip(*list_of_tuples)]

def flatten_rec(l):
    # TODO: fix problem with long lists (maximum recursion depth exceeded)
    if not l:
        return []
    if isinstance(l[0], list):
        return flatten(l[0]) + flatten(l[1:])
    return l[:1] + flatten(l[1:])

def flatten(l):
    """Flatten 2D lists"""
    return [i for sublist in l for i in sublist]

def remove_repeated_elements(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_uniq_elems(corpus):
    return list(set(flatten(corpus)))

##############################################################
# Input and output functions
##############################################################

def read_file_as_list_of_tuples(filename, delimiter='\t'):
    """It returns a list of tweets, and each tweet is a tuple of the elements found in the line"""
    with open(filename) as stream:
        reader = csv.reader(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        return [list(tuple(e) for e in g) for k, g in groupby(reader, lambda x: not x) if not k]

def read_file_as_lists(filename, delimiter='\t'):
    with open(filename) as stream:
        reader = csv.reader(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        labeled_tokens = [zip(*g) for k, g in groupby(reader, lambda x: not [s for s in x if s.strip()]) if not k]
        tokens, labels = zip(*labeled_tokens)
        return [list(t) for t in tokens], [list(l) for l in labels]

def read_test_tweets(filename, delimiter='\t'):
    with open(filename) as stream:
        reader = csv.reader(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        tweets = [list(g) for k, g in groupby(reader, lambda x: not [s for s in x if s.strip()]) if not k]
        return [flatten(tokens) for tokens in tweets]
    
def read_embeddings(fname, term2index, index2term, embeddings, sep=' ', skipfirst=False):
    with open(fname) as stream:
        if skipfirst:
            next(stream)
        for line in stream:
            # print(line)
            term2vec = line.strip().split(sep)
            term2index[term2vec[0]] = len(term2index)
            index2term.append(term2vec[0])
            embeddings = np.append(embeddings, np.array([term2vec[1:]], dtype=np.float32), axis=0)
        return term2index, index2term, embeddings
    
def pick_embeddings_from_file(filename, vocabulary, sep=' ', skipfirst=False):
    embeddings = []
    index2term = []
    reduced_vocab = {v:i for i,v in enumerate(list(vocabulary))} # To avoid removing elements from the outer list
    with open(filename) as stream:
        if skipfirst:
            next(stream)
        for line in stream:
            term2vec = line.strip().split(sep)
            if term2vec[0] in reduced_vocab:
                del reduced_vocab[term2vec[0]] # reduce the lookup space
                index2term.append(term2vec[0]) 
                embeddings.append(np.array(term2vec[1:], dtype=np.float32))
            if not reduced_vocab:
                break
    return index2term, np.array(embeddings)

def pick_embeddings(vocabulary, index2term, embeddings):
    new_embeddings = []
    new_index2term = []
    reduced_vocab = list(vocabulary) # To avoid removing elements from the outer list
    for index,term in enumerate(index2term):
        if term in reduced_vocab:
            reduced_vocab.pop(reduced_vocab.index(term)) # reduce the lookup space
            new_index2term.append(term) 
            new_embeddings.append(embeddings[index])
        if not reduced_vocab:
            break
    return new_index2term, np.array(new_embeddings)
    
def pick_embeddings_by_indexes(vocabulary, embeddings, term2index):
    embeds, index2term = zip(*[(embeddings[term2index.get(token)], token) 
                               for token in vocabulary 
                               if term2index.get(token)])
    return list(index2term), np.array(embeds)

def left_join_embeddings(vocab, ind2word_1, ind2word_2, embeddings_1, embeddings_2):
    embeddings = []
    index2word = []
    for word in vocab:
        if word in ind2word_1:
            index2word.append(word)
            embeddings.append(embeddings_1[ind2word_1.index(word)])
        elif word in ind2word_2:
            index2word.append(word)
            embeddings.append(embeddings_2[ind2word_2.index(word)])
    return index2word, np.array(embeddings)

def write_file(filename, dataset, delimiter='\t'):
    """dataset is a list of tweets where each token can be a tuple of n elements"""
    with open(filename, '+w') as stream:
        writer = csv.writer(stream, delimiter=delimiter, quoting=csv.QUOTE_NONE, quotechar='')
        for tweet in dataset:
            writer.writerow(list(tweet))

def save_file(filename, tweets, labels):
    """save a file with token, label and prediction in each row"""
    dataset = []
    for n, tweet in enumerate(tweets):
        tweet_data = list(zip(tweet, labels[n])) + [()]
        dataset += tweet_data 
    write_file(filename, dataset)

def write_encoded_tweets(filename, tweets):
    dataset = []
    for n, tweet in enumerate(tweets):
        dataset += tweet + [()]
    write_file(filename, dataset)
    
def save_predictions(filename, tweets, labels, predictions):
    """save a file with token, label and prediction in each row"""
    dataset, i = [], 0
    for n, tweet in enumerate(tweets):
        tweet_data = list(zip(tweet, labels[n], predictions[i:i + len(tweet)]))
        i += len(tweet)
        dataset += tweet_data + [()]
    write_file(filename, dataset)

def save_final_predictions(filename, tweets, predictions):
    """save a file with token and its prediction in each row"""
    dataset, i = [], 0
    for n, tweet in enumerate(tweets):
        tweet_data = list(zip(tweet, predictions[i:i + len(tweet)]))
        i += len(tweet)
        dataset += tweet_data + [()]
    write_file(filename, dataset)

def read_datasets():
    tweets_train, labels_train = read_file_as_lists(TRAIN_PREPROC_URL)
    tweets_dev,   labels_dev   = read_file_as_lists(DEV_PREPROC_URL)
    tweets_test,  labels_test  = read_file_as_lists(TEST_PREPROC_URL)

    # Combining train and dev to account for different domains
    tweets_train += tweets_dev
    labels_train += labels_dev

    return (tweets_train, labels_train), (tweets_test, labels_test)


def read_and_sync_postags(tweets_train, tweets_test):
    pos_tweets_train, pos_labels_train = read_file_as_lists(TRAIN_PREPROC_URL_POSTAG)
    pos_tweets_dev,   pos_labels_dev   = read_file_as_lists(DEV_PREPROC_URL_POSTAG)
    pos_tweets_test,  pos_labels_test  = read_file_as_lists(TEST_PREPROC_URL_POSTAG)

    # Combining train and dev to account for different domains
    pos_tweets_train += pos_tweets_dev
    pos_labels_train += pos_labels_dev

    # Standarizing tokenization between postags and original tweets
    sync_postags_and_tweets(tweets_train, pos_tweets_train, pos_labels_train)
    sync_postags_and_tweets(tweets_test, pos_tweets_test, pos_labels_test)

    return pos_labels_train, pos_labels_test


def read_twitter_embeddings(corpus):
    w2v_model = Word2Vec.load_word2vec_format(W2V_TWITTER_EMB_GODIN, binary=True)
    w2v_vocab = {token: v.index for token, v in w2v_model.vocab.items()}

    # Using only needed embeddings (faster this way)
    index2word, embeddings = pick_embeddings_by_indexes(get_uniq_elems(corpus), w2v_model.syn0, w2v_vocab)

    index2word = [PAD_TOKEN, UNK_TOKEN] + index2word
    word2index = ddict(lambda: index2word.index(UNK_TOKEN), {w: i for i, w in enumerate(index2word)})
    embeddings = np.append(np.zeros((2, embeddings.shape[1])), embeddings, axis=0)

    return embeddings, word2index


def read_gazetteer_embeddings():
    gazetteers = read_file_as_list_of_tuples(GAZET_EMB_ONE_CHECK)[0]
    index2gaze, embeddings = zip(*[(data[0], data[1:]) for data in gazetteers])

    index2gaze = [UNK_TOKEN, PAD_TOKEN] + list(index2gaze)
    gaze2index = ddict(lambda: index2gaze.index(UNK_TOKEN), {g: i for i, g in enumerate(index2gaze)})
    embeddings = np.append(np.zeros((2, 6)), embeddings, axis=0)

    return embeddings, gaze2index


def show_training_loss_plot(hist):
    # TODO: save the resulting plot
    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    plt.plot(range(len(train_loss)), train_loss, color="red", label="Train Loss")
    plt.plot(range(len(train_loss)), val_loss, color="blue", label="Validation Loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(loc="best")
    plt.show()

##############################################################
# Representation utilities
##############################################################

def tokenize_tweets(tweets, char_level=False, lower=False, filters=''):
    tokenizer = Tokenizer(filters=filters, lower=lower, char_level=char_level)
    tokenizer.fit_on_texts([' '.join(t) for t in tweets])
    return tokenizer

def get_max_word_length(tweets):
    return max(map(len, flatten(tweets)))

def element2index_dict(elems, offset=0):
    counter = Counter(elems) 
    sorted_elems = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return dict({e[0]: (i+offset) for i,e in enumerate(sorted_elems)})

def encode_tokens(token2index, tokens):
    return [[token2index[tkn] for tkn in tkns] for tkns in tokens]

def decode_predictions(predictions, idx2label):
    return [idx2label[pred] for pred in predictions]

def build_x_matrix(w, encodings, pad_idx=0):
    """
    w: window size for context
    encodings: list of lists, i.e. each tweet contains a list of tokens;
    return matrix: whose rows (of length w * 2 + 1) represent the tokens
    """
    x_matrix = []
    for context in encodings:
        for i, enc in enumerate(context):
            # Left side of the target word
            lower = max(i - w, 0)
            left = [pad_idx] * (w - (i - lower)) + context[lower:i]

            # Right side of the target word
            upper = min(i + w + 1, len(context))
            right = context[i:upper] + [pad_idx] * (w + 1 - (upper - i))

            # The whole vector (row)
            x_matrix.append(left + right)
    return np.array(x_matrix)

def build_sided_matrix(w, side, encodings, pad_idx=0):
    """
    side can be either 'left' or 'right'
    """
    x_matrix = []
    for context in encodings:
        for i, enc in enumerate(context):
            # Left side of the target word
            if side == 'left':
                lower = max(i - w, 0)
                left = [pad_idx] * (w - (i - lower)) + context[lower:i+1]
                x_matrix.append(left)
                
            # Right side of the target word
            elif side == 'right':
                upper = min(i + w + 1, len(context))
                right = context[i:upper] + [pad_idx] * (w + 1 - (upper - i))
                x_matrix.append(right)
    return np.array(x_matrix)


def vectorize_labels(labels, index2label=None):
    """labels: list of lists, i.e. each tweet has a list of labels"""
    flat_labels = flatten(labels)
    if not index2label:
        label_set = list(set(flat_labels))
        index2label = dict(enumerate(label_set))
    label2index = dict((l, i) for i, l in enumerate(index2label))

    y = [label2index[l] for l in flat_labels]
    y = to_categorical(np.array(y, dtype='int32'))
    return y, label2index, index2label

def build_embedding_matrix(w2v, dim, tokenizer):
    print("Length of word_index:", len(tokenizer.word_index))
    embedding_matrix = [w2v.syn0norm[w2v.vocab.get(word).index, :dim]
                        if w2v.vocab.get(word)
                        else np.zeros(dim)
                        for word, i in tokenizer.word_index.items()]
    print(len(embedding_matrix))
    print(embedding_matrix)
    return np.array(embedding_matrix, dtype='float32')

def orthigraphic_char(ch):
    try:
        if re.match('[a-z]', ch):
            return 'c'
        if re.match('[A-Z]', ch):
            return 'C'
        if re.match('[0-9]', ch):
            return 'n'
        if ch in string.punctuation:
            return 'p'
    except TypeError:
        print('TypeError:',ch)
    return 'x'
    
def orthographic_tweet(tweet):
    return [''.join([orthigraphic_char(ch) for ch in token]) for token in tweet]
    
def orthographic_mapping(tweets):
    return [orthographic_tweet(tweet) for tweet in tweets]

def match_up_to(x, elems):
    acc = [] 
    for e in elems: 
        acc.append(e)
        if x == ''.join(acc):
            return len(acc)
    return None
        
def map_equivalent(t):
    equivalences = [('&lt;', '<'), ('&amp;', '&'), ('&gt;', '>'),('&quot;', "\"")]
    for a, b in equivalences:
        if a in t:
            return t.replace(a, b)
    return t

def map_to_iob_labels(labels):
    return [[lbl[0] for lbl in lbls] for lbls in labels]

def sync_postags_and_tweets(tweets, pos_tweets, pos_labels):
    for row in range(len(tweets)):
        for pos in range(len(tweets[row])):
            t = tweets[row][pos]
            p = pos_tweets[row][pos] 

            if t != p:
                t = map_equivalent(t)
                up_to = match_up_to(t, pos_tweets[row][pos:])
                
                assert up_to and up_to > 0, "Inconsistency: {} not in {}".format(p, t)
                
                pos_chunk = remove_repeated_elements(pos_labels[row][pos:pos+up_to])
                
                del pos_tweets[row][pos+1:pos+up_to]
                del pos_labels[row][pos+1:pos+up_to]
                
                pos_tweets[row][pos] = t
                pos_labels[row][pos] = ''.join(pos_chunk)
                tweets[row][pos] = t
                
        assert len(tweets[row]) == len(pos_tweets[row]), "\n{}\n{}".format(tweets[row], pos_tweets[row])
        assert len(pos_tweets[row]) == len(pos_labels[row]), "{}\n{}".format(pos_tweets[row], pos_labels[row])


##############################################################
# Metrics functions
##############################################################
def fbeta_score(y_true, y_pred, beta=1):
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Weight precision and recall together as a single scalar.
    beta2 = beta ** 2
    f_score = (1 + beta2) * (precision * recall) / (beta2 * precision + recall)
    return f_score




