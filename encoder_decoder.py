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
from attention import Attention


class EncoderDecoder(Model):
    def __init__(self,
                 output_dim,
                 ):
        self.output_dim = output_dim
        super().__init__()
        self.encoder = Encoder(output_dim)
        self.decoder = Decoder(output_dim)

    def call(self, x):

        batch_size = int(x.shape[0])

        enc_out, enc_states = self.encoder(x)
        print('enc_out, enc_states: ', enc_out, enc_states)

        
        # output = layers.Input(batch_shape=(batch_size, 1, self.output_dim))
        output = K.zeros((batch_size, 1, self.output_dim))
        print('output: ', output, type(batch_size))
        dec_states = None

        for t in range(x.shape[1]):
            
            dec_out_t, dec_states = self.decoder(enc_out, prev_state_h=dec_states)
            dec_out_t = layers.Lambda(lambda x: K.reshape(x, (batch_size, 1, self.output_dim)))(dec_out_t)
            # print('dec_out_t: ', dec_out_t, output)
            output = layers.Concatenate(axis=1)([output, dec_out_t])

        print('dec_out_t: ', output)
        return layers.Lambda(lambda x: x[:, 1: , :])(output)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


class Encoder(Model):
    def __init__(self,
                 output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.lstm = layers.Bidirectional(LSTM(output_dim//2, return_state=True, return_sequences=True))

    def call(self, x):
        
        enc_output, forward_h, forward_c , backward_h, backward_c= self.lstm(x)
        enc_states = layers.Concatenate()([forward_h, backward_h])

        return [enc_output, enc_states]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[1], self.output_dim), (input_shape[0], self.output_dim)]


class Decoder(Model):
    def __init__(self,
                 output_dim):
        self.output_dim = output_dim
        super().__init__()

        self.lstm = LSTM(int(output_dim), return_state=True, return_sequences=True)
        self.attn = Attention(output_dim)

    def call(self, enc_out, prev_state_h):
   
        context_vector = self.attn(enc_out, prev_state_h=prev_state_h)
        dec_out_t, state_h, state_c = self.lstm(context_vector)
        # print('dec_out_t: ', dec_out_t)
        
        return [dec_out_t, state_h]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], 1, self.output_dim), (input_shape[0], self.output_dim)]
      

x = layers.Input(batch_shape=(8, 32, 1280))
y = EncoderDecoder(output_dim=64)(x)

model = Model(x, y)
print('Model: ', model.output.shape)

