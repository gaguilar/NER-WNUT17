import pycrfsuite as crf

from pycrfsuite import ItemSequence
from common import utilities as utils


def _get_xseq(model, matrix):
    xseq = [{'feat{}'.format(i):float(w) for i,w in enumerate(list(features))}
            for features
            in model.predict(matrix)]
    return ItemSequence(xseq)


def train_with_fextractor(nn_model, x_train, y_train):
    # nn_model = Model(inputs=model.input, outputs=model.get_layer('common_dense_layer').output)
    # x_train = [x_gaze_train,
    #            x_ortho_twitter_train,
    #            x_word_twitter_train,
    #            x_postag_train]

    xseq_train = _get_xseq(nn_model, x_train)
    yseq_train = utils.flatten(y_train)

    trainer = crf.Trainer(verbose=False)
    trainer.append(xseq_train, yseq_train)
    trainer.set_params({
        'c1': 1.0,                            # L1 penalty
        'c2': 1e-3,                           # L2 penalty
        'max_iterations': 100,                # stop earlier
        'feature.possible_transitions': True  # possible transitions, but not observed
    })
    trainer.train('weights.pycrfsuite')


def predict_with_fextractor(nn_model, x_test):
    # x_test = [x_gaze_test,
    #           x_ortho_twitter_test,
    #           x_word_twitter_test,
    #           x_postag_test]

    tagger = crf.Tagger()
    tagger.open('weights.pycrfsuite')

    # Predicting test data
    decoded_predictions = tagger.tag(_get_xseq(nn_model, x_test))
    return decoded_predictions






