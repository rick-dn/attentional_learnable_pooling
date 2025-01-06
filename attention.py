import random
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Layer
from keras.models import Model
from keras import activations
from keras.layers \
    import Dense, Embedding, LSTM, GRU, TimeDistributed, Input
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
import keras.backend as K
# from keras_self_attention import SeqSelfAttention
from keras.layers import add, concatenate


class Attention(Layer):
    '''
    Reference:
        "Effective Approaches to Attention-based Neural Machine Translation"
        https://arxiv.org/abs/1508.04025
    '''
    def __init__(self,
                 output_dim,  # suppose dim(hs) = dim(ht)
                 **kwargs):
        self.output_dim = output_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W_a = self.add_weight(name='W_a',
                                   shape=(input_shape[0], 1,
                                          self.output_dim),
                                   initializer='he_normal',
                                   trainable=True)

        # self.W_c = self.add_weight(name='W_c',
        #                            shape=(self.hidden_dim + self.hidden_dim,
        #                                   self.output_dim),
        #                            initializer='glorot_uniform',
        #                            trainable=True)

        # self.b = self.add_weight(name='b',
        #                          shape=(self.output_dim),
        #                          initializer='zeros',
        #                          trainable=True)


        super(Attention, self).build(input_shape)

    def call(self, x, prev_state_h):
        '''
        x: encoder output
        prev_state_h: decoder prev hidden state
        '''

        # print('x: ', x)
        # print('W_a: ', self.W_a)
        # print('prev_state_h: ', prev_state_h)
        
        if prev_state_h is not None:
           e_i_j = layers.Multiply()([x, prev_state_h])
        else:
           e_i_j = layers.Multiply()([x, self.W_a])
           

        attn_vec_i_j = activations.softmax(e_i_j, axis=1)

        context_vec_i_j = layers.Multiply()([x, attn_vec_i_j])

        context_vec_i =  layers.Lambda(lambda x: K.sum(x, axis=1, keepdims=True))(context_vec_i_j)
     
        return context_vec_i

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, self.output_dim)

    def compute_mask(self, inputs, mask):
        return mask
