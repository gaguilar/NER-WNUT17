import numpy as np
seed_number = 1337
np.random.seed(seed_number)

from common import utilities as utils
from common import representation as rep
from models import network
from models import crf
from settings import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def main():
    ###############################
    ## LOADING DATA
    ###############################

    ## TWEETS
    (tweets_train, labels_train), (tweets_test, labels_test) = utils.read_datasets()

    ## POS TAGS
    postag_train, postag_test = utils.read_and_sync_postags(tweets_train, tweets_test)


    ###############################
    ## LOADING EMBEDDINGS
    ###############################

    ## TWITTER
    twitter_embeddings, word2index = utils.read_twitter_embeddings(tweets_train + tweets_test)

    ## GAZETTERS
    gaze_embeddings, gaze2index = utils.read_gazetteer_embeddings()


    ###############################
    ## GENERATING ENCODING
    ###############################

    ## WORDS (X)
    radius = 1
    x_word_twitter_train = rep.encode_tweets(word2index, tweets_train, radius)
    x_word_twitter_test  = rep.encode_tweets(word2index, tweets_test, radius)

    ## LABELS (Y)
    y_bin_train = rep.encode_bin_labels(labels_train)
    y_cat_train = rep.encode_cat_labels(labels_train)

    ## POS TAGS
    index2postag = [PAD_TOKEN] + utils.get_uniq_elems(postag_train + postag_test)
    x_postag_train = rep.encode_postags(index2postag, postag_train, radius)
    x_postag_test  = rep.encode_postags(index2postag, postag_test, radius)

    ## ORTHOGRAPHY
    ortho_dim = 30
    ortho_max_length = 20
    x_ortho_train = rep.encode_orthography(tweets_train, ortho_max_length)
    x_ortho_test  = rep.encode_orthography(tweets_test, ortho_max_length)

    ## GAZETTEERS
    x_gaze_train = rep.encode_gazetteers(gaze2index, tweets_train, radius)
    x_gaze_test  = rep.encode_gazetteers(gaze2index, tweets_test, radius)


    ###############################
    ## BUILD NEURAL NETWORK
    ###############################

    char_inputs, char_encoded = network.get_char_cnn(ortho_max_length, len(rep.index2ortho), ortho_dim, 'char_ortho')
    word_inputs, word_encoded = network.get_word_blstm(len(index2postag), twitter_embeddings, window=radius*2+1, word_dim=100)
    gaze_inputs, gaze_encoded = network.get_gazetteers_dense(radius*2+1, gaze_embeddings)

    mtl_network = network.build_multitask_bin_cat_network(len(rep.index2category),      # number of category classes
                                                          char_inputs, char_encoded,    # char component (CNN)
                                                          word_inputs, word_encoded,    # word component (BLSTM)
                                                          gaze_inputs, gaze_encoded)    # gazetteer component (Dense)
    mtl_network.summary()


    ###############################
    ## TRAIN NEURAL NETWORK
    ###############################

    train_word_values = [x_word_twitter_train, x_postag_train]
    train_char_values = [x_ortho_train]
    train_gaze_values = [x_gaze_train]

    x_train_samples = train_gaze_values + train_char_values + train_word_values
    y_train_samples = {'bin_output': y_bin_train, 'cat_output': y_cat_train}

    network.train_multitask_net_with_split(mtl_network, x_train_samples, y_train_samples)


    ###############################
    ## NN PREDICTIONS
    ###############################

    x_test = [x_gaze_test, x_ortho_test, x_word_twitter_test, x_postag_test]
    inputs = gaze_inputs + char_inputs + word_inputs

    decoded_predictions = network.predict(mtl_network, inputs, x_test, rep.index2category)

    print("Classification Report\n")
    print(classification_report(utils.flatten(labels_test), decoded_predictions))
    print()
    print()
    print("Confusion Matrix\n")
    print(confusion_matrix(utils.flatten(labels_test), decoded_predictions))

    # Saving predictions in format: token\tlabel\tprediction
    utils.save_predictions(NN_PREDICTIONS, tweets_test, labels_test, decoded_predictions)


    ###############################
    ## CRF PREDICTIONS
    ###############################

    fextractor = network.create_model_from_layer(mtl_network, layer_name='common_dense_layer')
    crf.train_with_fextractor(fextractor, x_train_samples, labels_train)

    decoded_predictions = crf.predict_with_fextractor(fextractor, x_test)

    print("Classification Report\n")
    print(classification_report(utils.flatten(labels_test), decoded_predictions))
    print()
    print()
    print("Confusion Matrix\n")
    print(confusion_matrix(utils.flatten(labels_test), decoded_predictions))

    # Saving predictions in format: token label prediction
    utils.save_predictions(CRF_PREDICTIONS, tweets_test, labels_test, decoded_predictions)


if __name__ == '__main__':
    main()



