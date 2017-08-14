from collections import defaultdict as ddict
from common import utilities as utils
from keras.preprocessing.sequence import pad_sequences
from settings import *
from sklearn.preprocessing import LabelBinarizer


# TODO: get labels from corpus for other tasks
index2category = [
    'B-corporation',
    'B-creative-work',
    'B-group',
    'B-location',
    'B-person',
    'B-product',
    'I-corporation',
    'I-creative-work',
    'I-group',
    'I-location',
    'I-person',
    'I-product',
    'O'
]

index2segment = ['B', 'I', 'O']
index2ortho = ['x', 'c', 'C', 'n', 'p']


#############################################################################################

def encode_cat_labels(labels):
    encoded_cat, _, _ = utils.vectorize_labels(labels, index2category)
    return encoded_cat


def encode_seg_labels(labels):
    iob_labels = utils.map_to_iob_labels(labels)
    encoded_seg, _, _ = utils.vectorize_labels(iob_labels, index2segment)
    return encoded_seg


def encode_bin_labels(labels):
    lb = LabelBinarizer()
    encoded_bin = [['TRUE' if label != 'O' else 'FALSE' for label in lbls] for lbls in labels]
    encoded_bin = lb.fit_transform(utils.flatten(encoded_bin))
    return encoded_bin


#############################################################################################

def encode_tweets(word2index, tweets, radius):
    encoded_words = [[word2index[token] for token in tweet] for tweet in tweets]
    encoded_matrix = utils.build_x_matrix(radius, encoded_words, word2index[PAD_TOKEN])
    return encoded_matrix


def encode_postags(index2postag, postags, radius):
    postag2index = {w: i for i, w in enumerate(index2postag)}
    encoded_postags = [[postag2index[token] for token in tweet] for tweet in postags]
    encoded_postags = utils.build_x_matrix(radius, encoded_postags, postag2index[PAD_TOKEN])
    return encoded_postags


def encode_orthography(tweets, max_len):
    ortho2index = ddict(lambda: 0, {o: i for i, o in enumerate(index2ortho)})
    encoded_ortho = utils.orthographic_mapping(tweets)
    encoded_ortho = utils.encode_tokens(ortho2index, utils.flatten(encoded_ortho))
    encoded_ortho = pad_sequences(encoded_ortho, maxlen=max_len)
    return encoded_ortho


def encode_gazetteers(gaze2index, tweets, radius):
    encoded_gazetteers = [[gaze2index[token] for token in tweet] for tweet in tweets]
    encoded_gazetteers = utils.build_x_matrix(radius, encoded_gazetteers, gaze2index[PAD_TOKEN])
    return encoded_gazetteers
