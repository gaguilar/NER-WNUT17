from common import utilities as utils

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.initializers import RandomUniform
from keras.layers import Bidirectional
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import LSTM
from keras.layers.merge import concatenate
from keras.layers.pooling import GlobalAveragePooling1D
from keras.models import Model
from keras.optimizers import Adamax

import numpy as np


################################################################################
def _get_input_layer(shape, name):
    return Input(shape=shape, dtype='int32', name='{}_input'.format(name))


def _pretrained_emb_layer(embeddings, input_layer, input_len, name):
    embed_layer = Embedding(embeddings.shape[0],
                            embeddings.shape[1],
                            input_length=input_len,
                            weights=[embeddings],
                            trainable=False,
                            name='{}_embed'.format(name))(input_layer)
    embed_layer = Dropout(0.5, name='{}_embed_dropout'.format(name))(embed_layer)
    return embed_layer


def _rand_unif_emb_layer(input_layer, input_dim, output_dim,
                         input_len, name, seed=1337):
    # The range of the distribution is suggested by He et al. (2015)
    uniform = RandomUniform(seed=seed,
                            minval=-np.sqrt( 3 / output_dim ),
                            maxval= np.sqrt( 3 / output_dim ))
    embed_layer = Embedding(input_dim=input_dim,
                            output_dim=output_dim,
                            input_length=input_len,
                            embeddings_initializer=uniform,
                            trainable=False,
                            name='{}_embed'.format(name))(input_layer)
    embed_layer = Dropout(0.5, name='{}_embed_dropout'.format(name))(embed_layer)
    return embed_layer

################################################################################
def add_conv_layers(embedded, name, filters=64, kernel_size=3, dense_units=32, convs=2):
    conv_net = embedded
    for _ in range(convs):
        conv_net = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(conv_net)
    conv_net = GlobalAveragePooling1D()(conv_net)
    conv_net = Dense(dense_units, activation='relu', name='{}_dense'.format(name))(conv_net)
    return conv_net


def add_blstm_layers(embedded, name):
    lstm = LSTM(100, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)
    embedded = Bidirectional(lstm, name='{}_blstm'.format(name))(embedded)
    embedded = Dropout(0.5, name='{}_blstm_dropout'.format(name))(embedded)
    return embedded

################################################################################
def get_char_cnn(char_max_len,
                 char_vocab_size,
                 char_dim=30,
                 name='char'):
    char_input = _get_input_layer((char_max_len,), name)
    char_embed = _rand_unif_emb_layer(char_input, char_vocab_size, char_dim, char_max_len, name)
    char_encoded = add_conv_layers(char_embed, name + '_encoded')
    return [char_input], char_encoded


def get_word_blstm(vocab_size, embeddings, window=3, word_dim=100):
    # Twitter pre-trained word embeddings
    twitter_input = _get_input_layer((window,), 'word_twitter')
    twitter_embed = _pretrained_emb_layer(embeddings, twitter_input, window, 'word_twitter')

    # POS Tag random uniform distribution embeddings
    postag_input = _get_input_layer((window,), 'word_postag')
    postag_embed = _rand_unif_emb_layer(postag_input, vocab_size, word_dim, window, 'word_postag')

    word_encoded = concatenate([twitter_embed, postag_embed], axis=2)
    word_encoded = add_blstm_layers(word_encoded, 'word_encoded')
    return [twitter_input, postag_input], word_encoded


def get_gazetteers_dense(gazet_max_len, embeddings, name='gazetteers'):
    gazetteer_input = _get_input_layer((gazet_max_len,), name)
    gazetteer_embed = _pretrained_emb_layer(embeddings,
                                            gazetteer_input,
                                            gazet_max_len,
                                            name)
    # TODO: compare results with and without the dense layer
    # gazetteer_embed = Dense(units=32, activation="relu", name='gazetteer_dense')(gazetteer_embed)
    gazetteer_dense = Flatten()(gazetteer_embed)
    return [gazetteer_input], gazetteer_dense

################################################################################
def build_multitask_bin_cat_network(nb_classes,
                                    char_inputs, char_encoded,
                                    word_inputs, word_encoded,
                                    gaze_inputs, gaze_encoded):
    network = concatenate([gaze_encoded, char_encoded, word_encoded], name='concat_layer')
    network = Dense(100, activation='relu', name='common_dense_layer')(network)

    bin_output = Dense(1, activation='sigmoid', name='bin_output')(network)
    cat_output = Dense(nb_classes, activation='softmax', name='cat_output')(network)

    network_inputs  = gaze_inputs + char_inputs + word_inputs
    network_outputs = [bin_output, cat_output]

    model = Model(inputs=network_inputs, outputs=network_outputs, name='ne_model')

    adamax = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adamax,
                  loss={'bin_output': 'binary_crossentropy',
                        'cat_output': 'categorical_crossentropy'},
                  loss_weights={'bin_output': 1.,
                                'cat_output': 1.},
                  metrics={'bin_output': [utils.fbeta_score, 'accuracy'],
                           'cat_output': [utils.fbeta_score, 'accuracy']})
    return model


def build_multitask_seg_cat_network(nb_classes,
                                    char_inputs, char_encoded,
                                    word_inputs, word_encoded,
                                    gaze_inputs, gaze_encoded):
    network = concatenate([gaze_encoded, char_encoded, word_encoded], name='concat_layer')
    network = Dense(100, activation='relu', name='common_dense_layer')(network)

    seg_output = Dense(3, activation='softmax', name='seg_output')(network)
    cat_output = Dense(nb_classes, activation='softmax', name='cat_output')(network)

    network_inputs  = gaze_inputs + char_inputs + word_inputs
    network_outputs = [seg_output, cat_output]

    model = Model(inputs=network_inputs, outputs=network_outputs, name='ne_model')

    adamax = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adamax,
                  loss={'seg_output': 'categorical_crossentropy',
                        'cat_output': 'categorical_crossentropy'},
                  loss_weights={'seg_output': 1.,
                                'cat_output': 1.},
                  metrics={'seg_output': [utils.fbeta_score, 'accuracy'],
                           'cat_output': [utils.fbeta_score, 'accuracy']})
    return model


def build_multitask_bin_seg_cat_network(nb_classes,
                                        char_inputs, char_encoded,
                                        word_inputs, word_encoded,
                                        gaze_inputs, gaze_encoded):
    network = concatenate([gaze_encoded, char_encoded, word_encoded], name='concat_layer')
    network = Dense(100, activation='relu', name='common_dense_layer')(network)

    bin_output = Dense(1, activation='sigmoid', name='bin_output')(network)
    seg_output = Dense(3, activation='softmax', name='seg_output')(network)
    cat_output = Dense(nb_classes, activation='softmax', name='cat_output')(network)

    network_inputs  = gaze_inputs + char_inputs + word_inputs
    network_outputs = [bin_output, seg_output, cat_output]

    model = Model(inputs=network_inputs, outputs=network_outputs, name='ne_model')

    adamax = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=adamax,
                  loss={'bin_output': 'binary_crossentropy',
                        'seg_output': 'categorical_crossentropy',
                        'cat_output': 'categorical_crossentropy'},
                  loss_weights={'bin_output': 1.,
                                'seg_output': 1.,
                                'cat_output': 1.},
                  metrics={'bin_output': [utils.fbeta_score, 'accuracy'],
                           'seg_output': [utils.fbeta_score, 'accuracy'],
                           'cat_output': [utils.fbeta_score, 'accuracy']})
    return model

################################################################################
def train_multitask_net(mtl_model,
                        x_train, y_train,
                        x_dev, y_dev,
                        epochs=150,
                        batch_size=500,
                        verbose=True):
    early_stopping = EarlyStopping(patience=10, verbose=1)
    hist = mtl_model.fit(x_train,
                         y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=verbose,
                         shuffle=True,
                         callbacks=[early_stopping],
                         validation_data=(x_dev, y_dev))
    return hist


def train_multitask_net_with_split(mtl_model,
                                   x_train, y_train,
                                   epochs=150,
                                   batch_size=500,
                                   valid_split=0.2,
                                   verbose=True):
    early_stopping = EarlyStopping(patience=10, verbose=1)
    hist = mtl_model.fit(x_train,
                         y_train,
                         batch_size=batch_size,
                         epochs=epochs,
                         verbose=verbose,
                         shuffle=True,
                         validation_split=valid_split,
                         callbacks=[early_stopping])
    return hist

################################################################################
def predict(mtl_model, network_inputs, test_samples, index2label, verbose=True):
    # Single-task Network
    network = Model(inputs=network_inputs,
                    outputs=mtl_model.get_layer('cat_output').output,
                    name='cat_model')
    # Predicting test data
    prediction_probs = network.predict(test_samples, batch_size=500, verbose=verbose)

    # Decoding predictions
    decoded_predictions = utils.decode_predictions([np.argmax(p) for p in prediction_probs], index2label)

    return decoded_predictions

################################################################################

def create_model_from_layer(model, layer_name='common_dense_layer'):
    return Model(inputs=model.input, outputs=model.get_layer(layer_name).output)


