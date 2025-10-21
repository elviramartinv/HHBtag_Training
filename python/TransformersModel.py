# Definition of the model of the signal selection NN
# This file is part of https://github.com/hh-italian-group/hh-bbtautau.

import tensorflow as tf
import numpy as np
import json
import ROOT

from tensorflow.keras.layers import Layer, Dense, Dropout, TimeDistributed, MultiHeadAttention, LayerNormalization, Embedding
from tensorflow.keras.models import Model

class StdLayer(Layer):
    def __init__(self, file_name, var_pos, n_sigmas, **kwargs):
        with open(file_name) as json_file:
            data_json = json.load(json_file)
        n_vars = len(var_pos)
        self.vars_std = [1] * n_vars
        self.vars_mean = [1] * n_vars
        self.vars_apply = [False] * n_vars
        for var, ms in data_json.items():
            pos = var_pos[var]
            self.vars_mean[pos] = ms['mean']
            self.vars_std[pos] = ms['std']
            self.vars_apply[pos] = True
        self.vars_mean = tf.constant(self.vars_mean, dtype=tf.float32)
        self.vars_std = tf.constant(self.vars_std, dtype=tf.float32)
        self.vars_apply = tf.constant(self.vars_apply, dtype=tf.bool)
        self.n_sigmas = n_sigmas

        super(StdLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(StdLayer, self).build(input_shape)

    def call(self, X, training=False):
        Y = tf.clip_by_value(( X - self.vars_mean ) / self.vars_std, -self.n_sigmas, self.n_sigmas)
        return tf.where(self.vars_apply, Y, X)

class ScaleLayer(Layer):
    def __init__(self, file_name, var_pos, interval_to_scale, **kwargs):
        with open(file_name) as json_file:
            data_json = json.load(json_file)
        self.a = interval_to_scale[0]
        self.b = interval_to_scale[1]
        n_vars = len(var_pos)
        self.vars_max = [1] * n_vars
        self.vars_min = [1] * n_vars
        self.vars_apply = [False] * n_vars
        for var, mm in data_json.items():
            pos = var_pos[var]
            self.vars_min[pos] = mm['min']
            self.vars_max[pos] = mm['max']
            self.vars_apply[pos] = True
        self.vars_min = tf.constant(self.vars_min, dtype=tf.float32)
        self.vars_max = tf.constant(self.vars_max, dtype=tf.float32)
        self.vars_apply = tf.constant(self.vars_apply, dtype=tf.bool)
        self.y = (self.b - self.a) / (self.vars_max - self.vars_min)

        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)

    def call(self, X, training=False):
        Y = tf.clip_by_value( (self.y * ( X - self.vars_min))  + self.a , self.a, self.b)
        return tf.where(self.vars_apply, Y, X)

class PositionalEncoding(Layer):
    def __init__(self, max_sequence_length=20, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_sequence_length = max_sequence_length
    
    def build(self, input_shape):
        super(PositionalEncoding, self).build(input_shape)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        
        # Create position indices
        position = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
        position = tf.expand_dims(position, axis=1)  # Shape: (seq_len, 1)
        
        # Create dimension indices for positional encoding
        i = tf.range(tf.cast(d_model, tf.float32), dtype=tf.float32)
        
        # Calculate the angle rates
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        
        # Apply sin to even indices and cos to odd indices
        angle_rads = position * angle_rates
        
        # Apply sin to even indices (0, 2, 4, ...)
        angle_rads_sin = tf.sin(angle_rads)
        # Apply cos to odd indices (1, 3, 5, ...)
        angle_rads_cos = tf.cos(angle_rads)
        
        # Create the final positional encoding by interleaving sin and cos
        # Use modulo to determine even/odd positions
        indices = tf.range(d_model, dtype=tf.int32)
        even_mask = tf.equal(indices % 2, 0)
        
        # Create positional encoding
        pe = tf.where(even_mask, angle_rads_sin, angle_rads_cos)
        
        # Add batch dimension: (seq_len, d_model) -> (1, seq_len, d_model)
        pe = tf.expand_dims(pe, axis=0)
        
        # Broadcast to batch size: (1, seq_len, d_model) -> (batch_size, seq_len, d_model)
        pe = tf.tile(pe, [batch_size, 1, 1])
        
        return inputs + pe

class HHModel(Model):
    def __init__(self, var_pos, mean_std_json, min_max_json, params):
        super(HHModel, self).__init__()

        self.normalize = StdLayer(mean_std_json, var_pos, 5, name='std_layer')
        self.scale = ScaleLayer(min_max_json, var_pos, [-1,1], name='scale_layer')

        # Embedding layer - projects input features to embedding dimension
        self.embedding_layer = TimeDistributed(Dense(params["embedding_dim"], activation='relu'), 
                                             name="embedding_layer")
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(name='positional_encoding')
        
        # Transformer layers
        self.transformer_layers = []
        for i in range(params.get('num_transformer_layers', 1)):
            layer_dict = {
                'attention': MultiHeadAttention(num_heads=params['num_attention_heads'],
                                              key_dim=params['embedding_dim'],
                                              dropout=params.get('dropout_rate_transformer', 0.1),
                                              name=f'multi_head_attention_{i}'),
                'norm1': LayerNormalization(name=f'layer_norm_1_{i}'),
                'feed_forward': TimeDistributed(Dense(params['feed_forward_dim'], activation='relu'),
                                              name=f'feed_forward_{i}'),
                'ff_projection': TimeDistributed(Dense(params['embedding_dim']),
                                               name=f'ff_projection_{i}'),
                'norm2': LayerNormalization(name=f'layer_norm_2_{i}'),
                'dropout1': Dropout(params.get('dropout_rate_transformer', 0.1), 
                                  name=f'dropout_1_{i}'),
                'dropout2': Dropout(params.get('dropout_rate_transformer', 0.1),
                                  name=f'dropout_2_{i}')
            }
            self.transformer_layers.append(layer_dict)

        # Projection back to original feature space (optional, for residual connection)
        if params['embedding_dim'] != (len(var_pos) - 1):  # -1 because we skip the mask feature
            self.projection_layer = TimeDistributed(Dense(len(var_pos) - 1), name='projection_layer')
        else:
            self.projection_layer = None

        # Post Dense Block
        self.dense_post = []
        self.dropout_dense_post = []
        self.batch_norm_dense_post = []
        for i in range(params['num_den_layers_post']):
            self.dense_post.append(TimeDistributed(Dense(params['num_units_den_layers_post'],
                                                         activation=params['activation_dense_post']),
                                                          name='dense_pos_{}'.format(i)))
            self.batch_norm_dense_post.append(tf.keras.layers.BatchNormalization(name='batch_normalization_post_{}'.format(i)))
            if params['dropout_rate_den_layers_post'] > 0:
                self.dropout_dense_post.append(Dropout(params['dropout_rate_den_layers_post'],
                                                       name='dropout_dense_post_{}'.format(i)))

        self.final_dense = TimeDistributed(Dense(1, activation="sigmoid"), name='output')

    @tf.function
    def call(self, inputs, training=False):
        # Preprocessing
        x = self.normalize(inputs, training=training)
        x = self.scale(x, training=training)
        
        # Create mask (jets with pt > 0.5 are valid)
        mask = tf.convert_to_tensor(inputs[:, :, 0] > 0.5, dtype=tf.bool)
        
        # Remove the mask feature and embed
        x_features = x[:, :, 1:]  # Skip the first feature (mask)
        x = self.embedding_layer(x_features, training=training)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply transformer layers
        for transformer_layer in self.transformer_layers:
            # Multi-head attention with residual connection
            # Remove attention_mask for now to fix dimension issues
            attn_output = transformer_layer['attention'](x, x, training=training)
            attn_output = transformer_layer['dropout1'](attn_output, training=training)
            x_residual = transformer_layer['norm1'](x + attn_output)
            
            # Feed-forward with residual connection
            ff_output = transformer_layer['feed_forward'](x_residual, training=training)
            ff_output = transformer_layer['ff_projection'](ff_output, training=training)  # Project back to embedding_dim
            ff_output = transformer_layer['dropout2'](ff_output, training=training)
            x = transformer_layer['norm2'](x_residual + ff_output)
        
        # Optional projection back to feature space for residual connection
        if self.projection_layer is not None:
            x_projected = self.projection_layer(x, training=training)
            x = x_projected + x_features  # Residual connection with original features
        else:
            x = x  # Keep transformer output
        
        # Post-processing dense layers
        for i in range(len(self.dense_post)):
            x = self.dense_post[i](x, training=training)
            if len(self.batch_norm_dense_post) > i:
                x = self.batch_norm_dense_post[i](x, training=training)
            if len(self.dropout_dense_post) > i:
                x = self.dropout_dense_post[i](x, training=training)
        
        # Final output layer
        x = self.final_dense(x, training=training)

        # Reshape and apply masking (same as LSTM model)
        input_shape = tf.shape(inputs)
        x = tf.reshape(x, shape=(input_shape[0], input_shape[1]))
        x = x * tf.cast(mask, dtype=tf.float32)
        s = tf.reshape(tf.reduce_sum(x, axis=1), shape=(input_shape[0], 1))
        x = (2 * x) / s
        return x

def ListToVector(files):
    v = ROOT.std.vector('string')()
    for file in files:
        v.push_back(file)
    return v

def sel_acc(y_true, y_pred, n_positions, n_exp, do_ratio, return_num=False):
    pred_sorted = tf.argsort(y_pred, axis=1, direction='DESCENDING')
    n_evt = tf.shape(y_true)[0]
    evt_id = tf.range(n_evt)
    matches_vec = []
    for n in range(n_positions):
        index = tf.transpose(tf.stack([evt_id, tf.reshape(pred_sorted[:, n], shape=(n_evt,))]))
        matches_vec.append(tf.gather_nd(y_true, index))
    matches_sum = tf.add_n(matches_vec)
    valid = tf.cast(tf.equal(matches_sum, n_exp), tf.float32)
    if do_ratio:
        n_valid = tf.reduce_sum(valid)
        ratio = n_valid / tf.cast(n_evt, tf.float32)
        if return_num:
            ratio = ratio.numpy()
        return ratio
    return valid

def sel_acc_2(y_true, y_pred):
    return sel_acc(y_true, y_pred, 2, 2, False)
def sel_acc_3(y_true, y_pred):
    return sel_acc(y_true, y_pred, 3, 2, False)
def sel_acc_4(y_true, y_pred):
    return sel_acc(y_true, y_pred, 4, 2, False)
