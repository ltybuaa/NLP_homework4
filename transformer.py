# -*- coding: gbk -*-
import re
import tensorflow as tf
import numpy as np

def is_uchar(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    if uchar in ('��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '��', '����'):
        return True
    return False


# ����������
def data_generator(data, batch_size, time_steps):
    num_batches = len(data) // (batch_size * time_steps)
    data = data[:num_batches * batch_size * time_steps]
    data = np.array(data).reshape((batch_size, -1))
    while True:
        for i in range(0, data.shape[1], time_steps):
            x = data[:, i:i + time_steps]
            y = np.roll(x, -1, axis=1)
            yield [x, y], y

# Transformer��������
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Transformer��������
class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

# ������Transformerģ��
class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.decoder_layers = [TransformerDecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training=False):
        inp, tar = inputs
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_output = self.encoder_embedding(inp)
        for i in range(len(self.encoder_layers)):
            enc_output = self.encoder_layers[i](enc_output, training, enc_padding_mask)
        dec_output = self.decoder_embedding(tar)
        for i in range(len(self.decoder_layers)):
            dec_output = self.decoder_layers[i](dec_output, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output

    def create_masks(self, inp, tar):
        enc_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask

    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask


# ����ص�����
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))



# �ı����ɺ���
def generate_text(model, start_string, num_generate=100):
    input_eval = [char2id[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    decoder_input = tf.expand_dims([char2id['��']], 0)  # ��ʼdecoder input

    for _ in range(num_generate):
        predictions = model([input_eval, decoder_input], training=False)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.concat([input_eval, tf.expand_dims([predicted_id], 0)], axis=-1)
        decoder_input = tf.concat([decoder_input, tf.expand_dims([predicted_id], 0)], axis=-1)
        text_generated.append(id2char[predicted_id])

    return start_string + ''.join(text_generated)


if __name__ == "__main__":
    # ��ȡ�ʹ����ı�
    with open(r'�����˲�.txt', encoding='gbk', errors='ignore') as f:
        data = f.readlines()

    pattern = re.compile(r'\(.*\)')
    data = [pattern.sub('', lines) for lines in data]
    data = [line.replace('����', '��') for line in data if len(line) > 1]
    data = ''.join(data)
    data = [char for char in data if is_uchar(char)]
    data = ''.join(data)

    # �����ʻ��
    vocab = list(set(data))
    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = {i: c for i, c in enumerate(vocab)}
    numdata = [char2id[char] for char in data]
    # ���ó�����
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    input_vocab_size = len(vocab) + 2
    target_vocab_size = len(vocab) + 2
    dropout_rate = 0.1

    batch_size = 32
    time_steps = 50
    epochs = 50
    learning_rate = 0.001

    model = TransformerModel(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, time_steps,
                             time_steps, dropout_rate)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    train_data = data_generator(numdata, batch_size, time_steps)

    history = LossHistory()

    # ѵ��ģ��
    model.fit(train_data, epochs=epochs, steps_per_epoch=len(numdata) // (batch_size * time_steps), callbacks=[history])

    # �����ı�ʾ��
    print(generate_text(model, start_string="���׽��������������Ǳ�����ʿ�������е��Ƕ������������˹�ͬ��������Ĺ�֤�ˣ���������ǰ������ļα�"))