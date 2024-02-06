
with open('input.txt', 'r',encoding='utf-8') as f:
    text = f.read()

print('length of dataset in characters: ',len(text))
#|%%--%%| <FEtQ4Ve4Yj|alqfF0uaQD>

# Lets look at the first 200 words
print(text[:200])

#|%%--%%| <alqfF0uaQD|Fc3Z1zOsmq>

# What kind of dictionary are we working with?
# notice that there is a space character
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# these are the list of characters that the model can see and emit

#|%%--%%| <Fc3Z1zOsmq|CN5o2XhN1d>
# Tokenization
# Time to encode characters into integers
# CHALLENGE: Implement another tokenizer, why is it better? (sub-word unit, byte-pair)
    # Explain the trade-off of large vocabulary size and length of sequences

# create an elementary mapping from characters to integers
# simple uses integer ordering to create encoding
stoi =  { ch:i for i,ch in enumerate(chars)}
# and it's inverse mapping
iots = { i:ch for i,ch in enumerate(chars)}

# define lambda functions that use mappings
encode = lambda s: [stoi[c] for c in s]
decode = lambda t: [''.join(iots[c] for c in t)]

print(encode('Your are the first citizen'))
print(decode(encode('Your are the first citizen')))

#|%%--%%| <CN5o2XhN1d|vbi7wpwLzY>

import torch

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:500]) # first 500 characters again, but this time tokenized

#|%%--%%| <vbi7wpwLzY|EAhfTXPDG8>

# Creating Train and Validation Split

n = int(0.9*len(data)) #first 90% of data
train_data = data[:n]
val_data = data[n:]

#|%%--%%| <EAhfTXPDG8|vcTD1LVK0D>

# for training we feed in chunks of text
# we call this block_size, or context_length

block_size = 8
train_data[:block_size+1] # block_size+1 because we start off including the first integer (given [15,35,25,...] , from 15, 35 can come next. From 15,35 ,25 comes next. All the way to block_size+1, which gives 8 examples


#|%%--%%| <vcTD1LVK0D|7TpihwOfbo>
# illustration of how nn is trained simultaneously and why we take block_size+1
# 8 examples in the chunk
x=train_data[:block_size]
y=train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")
# this also gets the transformer used to seeing a wide range of context (1 character to block size)
#|%%--%%| <7TpihwOfbo|TNlgVyIuBE>

torch.manual_seed(1337)
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) #stacks 1D tensors on top of each other
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")

