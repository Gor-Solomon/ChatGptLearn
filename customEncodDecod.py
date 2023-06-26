import torch

def get_custom_encoder(characters):
    stoi = {ch: i for i, ch in enumerate(characters)}

    def encoder(text):
        return [stoi[c] for c in text]

    return encoder


def get_custom_decoder(characters):
    itos = {i: ch for i, ch in enumerate(characters)}

    def decoder(index):
        return [itos[i] for i in index]

    return decoder


def batch_factory(training_data, validation_data, batch_size, block_size):
    def batch(split):
        data = training_data if split == 'training' else validation_data
        # generates batch_size numbers which are randomly selected from the data set -block data so we don't go out
        # of array range
        xi = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in xi])
        y = torch.stack([data[i+1:i+block_size+1] for i in xi])
        return x, y

    return batch



'''
shift_size = 1
x = train_data[:block_size]
y = train_data[shift_size:block_size+shift_size]

for i in range(block_size):
    context = x[:i+1]
    target = y[i]
    print(f'x = {context}, y = {target}')
'''
