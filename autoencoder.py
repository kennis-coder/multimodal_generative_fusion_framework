# -*- encoding: utf-8 -*-

# @Date    : 1/4/20
# @Author  : Kennis Yu

from keras.layers import (Input, Embedding, LSTM, Dense,
                          RepeatVector, TimeDistributed, Dot, Activation, Flatten, Lambda)
from keras.models import Model
# from sklearn.preprocessing import minmax_scale
# from keras.utils import plot_model
from keras import backend as K
from .utils import load_embedding_matrix
from .utils import Word2Embedded, tanh3
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2, 3'


class EncodingGenerator(object):
    def __init__(self, texts_size=400, embedding_size=300, vocab_size=60230,
                 texts_autoencoder_units=100, leaves_size=300,
                 leaves_autoencoder_units=10, texts_att_size=60,
                 leaves_att_size=50):
        # texts autoencoder
        embedding_matrix = load_embedding_matrix()
        texts_in = Input(shape=(texts_size,), dtype='int32', name='texts_in')
        embedded_texts = Embedding(output_dim=embedding_size, input_dim=vocab_size,
                                   input_length=texts_size, weights=[embedding_matrix],
                                   trainable=True)(texts_in)

        texts_encoder_out = LSTM(texts_autoencoder_units, activation='tanh',
                                 input_shape=(texts_size, embedding_size))(embedded_texts)
        self.texts_encoder = Model(inputs=texts_in, outputs=texts_encoder_out)
        self.texts_encoder.compile(optimizer='adam', loss='mse')

        hidden = RepeatVector(texts_size)(texts_encoder_out)
        hidden = LSTM(texts_autoencoder_units, activation='tanh', return_sequences=True)(hidden)
        texts_out = TimeDistributed(Dense(units=embedding_size, activation=tanh3),
                                    name='texts_decoder_out')(hidden)

        # leaves autoencoder
        leaves_in = Input(shape=(leaves_size,), dtype='float32', name='leaves_in')
        leaves_encoder_out = Dense(units=leaves_autoencoder_units,
                                   activation='tanh')(leaves_in)
        self.leaves_encoder = Model(inputs=leaves_in, outputs=leaves_encoder_out, name='leaves_encoder')
        self.leaves_encoder.compile(optimizer='adam', loss='mse')

        leaves_out = Dense(units=leaves_size, activation='linear', name='leaves_decoder_out')(leaves_encoder_out)

        # attention-mechanism based translation
        key = Dense(units=texts_att_size, name='key')(texts_encoder_out)
        key = Lambda(lambda x: K.expand_dims(x, axis=-1))(key)

        value = Dense(units=texts_att_size, name='value')(texts_encoder_out)
        value = Lambda(lambda x: K.expand_dims(x, axis=-1))(value)

        query = Dense(units=leaves_att_size, name='query')(leaves_encoder_out)
        query = Lambda(lambda x: K.expand_dims(x, axis=-1))(query)

        attention = Dot(axes=[2, 2], normalize=True)([query, key])
        attention = Activation(activation='softmax', name='attention_weight')(attention)
        context = Dot(axes=[2, 1])([attention, value])
        context = Flatten()(context)
        gen_leaves = Dense(units=leaves_size, activation='linear', name='gen_leaves')(context)

        # the whole model
        self.whole_model = Model(inputs=[texts_in, leaves_in], outputs=[texts_out, leaves_out, gen_leaves],
                                 name='whole_model')
        self.whole_model.compile(optimizer='adam', loss='mse',
                                 loss_weights={'gen_leaves': 0.5, 'texts_decoder_out': 0.25,
                                               'leaves_decoder_out': 0.25})

        # plot_model(self.whole_model, show_shapes=True, to_file='whole_model.png')

    def training(self, texts, leaves):
        word2embedded = Word2Embedded(400)
        word_embedding = word2embedded(texts)
        self.whole_model.fit(x={'texts_in': texts, 'leaves_in': leaves},
                             y={'texts_decoder_out': word_embedding,
                                'leaves_decoder_out': leaves,
                                'gen_leaves': leaves},
                             epochs=3
                             )

    def encoding_texts(self, texts):
        return self.texts_encoder.predict(x={'texts_in': texts})

    def encoding_leaves(self, leaves):
        return self.leaves_encoder.predict(x={'leaves_in': leaves})
