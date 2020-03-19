# -*- encoding: utf-8 -*-

# @Date    : 1/4/20
# @Author  : Kennis Yu

from keras.layers import (Input, Embedding, LSTM, Dense,
                          RepeatVector, TimeDistributed,
                          Dot, Activation,
                          Lambda, Multiply, Softmax, Conv2D,
                          MaxPool2D, Flatten, BatchNormalization)
from keras.callbacks import EarlyStopping, TerminateOnNaN
from numpy import asarray
from sklearn.externals import joblib
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
from keras.losses import mse, binary_crossentropy
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'
EMBEDDING_MATRIX_FN = "word2vec_embedding_matrix.pkl"


def read(fname):
    with open(fname, 'rb') as fr:
        return joblib.load(fr)


def load_embedding_matrix():
    embedding_matrix = asarray(read(EMBEDDING_MATRIX_FN), dtype="float32")
    return embedding_matrix


class Word2Embedded(object):
    def __init__(self, text_len):
        embedding_matrix = load_embedding_matrix()
        text_input = Input(shape=(text_len,), dtype="int32")
        embedded = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                             input_length=text_len, weights=[embedding_matrix], trainable=False)(text_input)
        self.model = Model(inputs=[text_input], outputs=[embedded])

    def __call__(self, texts):
        """

        :type texts: list.
        """
        return self.model.predict(texts)


class EncodingGenerator(object):
    def __init__(self, texts_size=400, embedding_size=300, vocab_size=60230,
                 texts_autoencoder_units=60,
                 leaves_size=300,
                 leaves_autoencoder_units=30,
                 conv2d_filters=10,
                 conv2d_kernel_size=3
                 ):
        # texts autoencoder
        embedding_matrix = load_embedding_matrix()
        texts_in = Input(shape=(texts_size,), dtype='int32', name='texts_in')
        embedded_texts = Embedding(output_dim=embedding_size, input_dim=vocab_size,
                                   input_length=texts_size, weights=[embedding_matrix],
                                   trainable=True)(texts_in)

        texts_encoder_out = LSTM(texts_autoencoder_units, activation='tanh',
                                 input_shape=(texts_size, embedding_size), name='texts_encoder_out')(embedded_texts)
        # texts_encoder_out = BatchNormalization()(texts_encoder_out)
        self.texts_encoder = Model(inputs=texts_in, outputs=texts_encoder_out)
        self.texts_encoder.compile(optimizer='adam', loss='mse')

        hidden = RepeatVector(texts_size)(texts_encoder_out)
        hidden = LSTM(texts_autoencoder_units, activation='relu', return_sequences=True)(hidden)
        texts_out = TimeDistributed(Dense(units=embedding_size), name='texts_decoder_out')(hidden)
        # texts_out = Flatten(name='texts_decoder_out')(texts_out)

        # leaves autoencoder
        leaves_in = Input(shape=(leaves_size,), dtype='float32', name='leaves_in')
        leaves_encoder_out = Dense(units=leaves_autoencoder_units,
                                   activation='relu', name='leaves_encoder_out')(leaves_in)
        leaves_encoder_bn = BatchNormalization()(leaves_encoder_out)
        self.leaves_encoder = Model(inputs=leaves_in, outputs=leaves_encoder_out, name='leaves_encoder')
        self.leaves_encoder.compile(optimizer='adam', loss='mse')

        leaves_out = Dense(units=leaves_size, activation='relu', name='leaves_decoder_out')(leaves_encoder_bn)

        # attention-mechanism based translation
        # key = Dense(units=texts_autoencoder_units, name='key')(texts_encoder_out)
        key = texts_encoder_out
        expanded_key = Lambda(lambda x: K.expand_dims(x, axis=-1))(key)

        # value = Dense(units=texts_autoencoder_units, name='value')(texts_encoder_out)
        value = texts_encoder_out
        repeated_value = RepeatVector(n=leaves_autoencoder_units)(value)

        # query = Dense(units=leaves_autoencoder_units, name='query')(leaves_encoder_out)
        query = leaves_encoder_bn
        expanded_query = Lambda(lambda x: K.expand_dims(x, axis=-1))(query)

        attention = Dot(axes=[2, 2], normalize=True)([expanded_query, expanded_key])
        attention = Activation(activation='softmax', name='attention_weight')(attention)
        self.attention_model = Model(inputs=[texts_in, leaves_in], outputs=[attention],
                                     name='attention_model')

        weighted_value = Multiply()([attention, repeated_value])
        context = Lambda(lambda x: K.sum(x, axis=-1, keepdims=False))(weighted_value)
        gen_leaves = Dense(units=leaves_size, activation='relu', name='gen_leaves')(context)

        # joint representation: cooccurrence matrix
        value_probability = Softmax(axis=-1, name='value_probability')(weighted_value)
        repeated_query = Lambda(lambda x: K.repeat_elements(x, rep=texts_autoencoder_units, axis=-1))(expanded_query)
        query_probability = Softmax(axis=1, name='query_probability')(repeated_query)
        cooccurrence = Multiply(name='cooccurrence')([value_probability, query_probability])

        # cooccurrence matrix is used for classification
        cooccurrence = Lambda(lambda x: K.expand_dims(x, axis=-1))(cooccurrence)
        hidden = Conv2D(filters=conv2d_filters, kernel_size=conv2d_kernel_size, strides=(1, 1),
                        padding='valid', dilation_rate=(2, 2), activation='relu')(cooccurrence)
        hidden = MaxPool2D()(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Conv2D(filters=conv2d_filters, kernel_size=conv2d_kernel_size, strides=(1, 1),
                        padding='valid', dilation_rate=(2, 2), activation='relu')(hidden)
        hidden = MaxPool2D()(hidden)
        hidden = BatchNormalization()(hidden)
        hidden = Flatten()(hidden)
        hidden = Dense(units=conv2d_filters, activation='relu')(hidden)
        hidden = BatchNormalization()(hidden)
        final_output = Dense(units=2, activation='softmax', name='final_output')(hidden)
        self.classification = Model(inputs=[texts_in, leaves_in], outputs=[final_output])
        self.classification.compile(optimizer='adam', loss=binary_crossentropy)

        # the whole model
        self.whole_model = Model(inputs=[texts_in, leaves_in],
                                 outputs=[texts_out, leaves_out, gen_leaves, final_output],
                                 name='whole_model')
        self.whole_model.compile(optimizer='adam', loss={'gen_leaves': mse, 'texts_decoder_out': mse,
                                                         'leaves_decoder_out': mse,
                                                         'final_output': binary_crossentropy},
                                 loss_weights={'gen_leaves': 0.5, 'texts_decoder_out': 0.25,
                                               'leaves_decoder_out': 0.25, 'final_output': 1.})

        plot_model(self.whole_model, show_shapes=True, to_file='whole_model.png')

    def training(self, texts, leaves, targets, epochs=1):
        word2embedded = Word2Embedded(400)
        word_embedding = word2embedded(texts)
        early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min", min_delta=0.0001)
        terminate_on_nan = TerminateOnNaN()
        self.whole_model.fit(x={'texts_in': texts, 'leaves_in': leaves},
                             y={'texts_decoder_out': word_embedding,
                                'leaves_decoder_out': leaves,
                                'gen_leaves': leaves,
                                'final_output': targets
                                },
                             validation_split=0.25,
                             shuffle=True,
                             callbacks=[early_stopping, terminate_on_nan],
                             epochs=epochs
                             )

    def encoding_texts(self, texts):
        return self.texts_encoder.predict(x={'texts_in': texts})

    def get_text_encoder_weights(self):
        return self.texts_encoder.get_layer(name='texts_encoder_out').get_weights()

    def encoding_leaves(self, leaves):
        return self.leaves_encoder.predict(x={'leaves_in': leaves})

    def get_leaf_encoder_weights(self):
        return self.leaves_encoder.get_layer(name='leaves_encoder_out').get_weights()

    def get_attention(self, texts, leaves):
        return self.attention_model.predict(x={'texts_in': texts, 'leaves_in': leaves})

    def predict(self, texts, leaves):
        return self.classification.predict(x={'texts_in': texts, 'leaves_in': leaves})


from numpy import concatenate
from keras.utils import to_categorical

texts = read('imbalanced_data/texts.pkl')
texts = concatenate(texts[:2])
leaves = read('imbalanced_data/leaves.pkl')
leaves = concatenate(leaves[:2])
targets = read('imbalanced_data/target.pkl')
targets = concatenate(targets[:2])
targets = to_categorical(targets)
eg = EncodingGenerator()
eg.training(texts, leaves, targets)
eg.predict(texts, leaves)