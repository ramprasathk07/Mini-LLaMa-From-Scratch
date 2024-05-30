import urllib.request
import os 
import torch 
import yaml
import logging

#-----------------------------------------------------------------------------------------------------------------------
log = 'Logs'
os.makedirs(log,exist_ok=True)
logging.basicConfig(filename=f'{log}/Data.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')
#-----------------------------------------------------------------------------------------------------------------------
#Download the data if required ~1M
file_name = "data/tinyshakespeare.txt"

if not os.path.exists(file_name):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    urllib.request.urlretrieve(url, file_name)

#Data Proceessing
lines = open(file_name,'r').read()

'''
vocab to store no of unique chars in the corpus
'''

with open('utils/config.yaml') as f:
    MASTER_CONFIG = yaml.safe_load(f)

vocab = sorted(list(set(lines)))
print('Total number of characters in our dataset (Vocabulary Size):', len(vocab))
logging.info(f'Total number of characters in our dataset (Vocabulary Size):{len(vocab)}')
#Create the mapping to no for embedding 

itos = {i:c for i,c in enumerate(vocab)}
stoi = {c:i for i,c in enumerate(vocab)}

def encode(s):
    return [stoi[ch] for ch in s]

def decode(l):
    return ''.join([itos[i] for i in l])

print(f"encode Morning:{encode('Morning')}")
print(f"decode {encode('Morning')}:{decode(encode('Morning'))}")

#create a dataset with all lines in corpus
dataset = torch.tensor(encode(lines), dtype=torch.int8)
print(f"Data shape:{dataset.shape}")
logging.info(f"Data shape:{dataset.shape}")

# Function to get batches for training, validation, or testing
def get_batches(split, batch_size = 8, context_window = 16, data = dataset):
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]

    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test

    # Pick random starting points within the data
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))

    # Create input sequences (x) and corresponding target sequences (y)
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window+1] for i in ix]).long()

    return x, y

# Obtain batches for training using the specified batch size and context window
xs, ys = get_batches(data = dataset, split='train')
print(f"\nxs:{xs}\n")
# Decode the sequences to obtain the corresponding text representations
decoded_samples = [(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]

# Print the random sample
print(decoded_samples)