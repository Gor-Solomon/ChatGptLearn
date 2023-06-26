import random
import numpy as np
import tensorflow as tf
import customEncodDecod
import torch

fileUrl = 'https://www.o-bible.com/download/kjv.txt'
fileName = 'full-bible.txt'
modelFileName = 'text_generator_full_bible.model'

filepath = tf.keras.utils.get_file(fileName, fileUrl)
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()

characters = sorted(list(set(text)))
vocab_size = len(characters)

encoder = customEncodDecod.get_custom_encoder(characters)
decoder = customEncodDecod.get_custom_decoder(characters)
data = torch.tensor(encoder(text), dtype=torch.long)

# splitting data into trained and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
validation_data = data[n:]
block_size = 8  # what is the maximum context length for predictions?
batch_size = 4  # how many independent sequences we process in parallel?

batch = customEncodDecod.batch_factory(train_data, validation_data, batch_size, block_size)
xb, yb = batch('training')
print(f'X = {xb.shape}')
print(f'X = {xb}')
print(f'Y = {yb.shape}')
print(f'Y = {yb}')
