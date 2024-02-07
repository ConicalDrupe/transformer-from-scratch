
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
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #moving window target
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

#|%%--%%| <TNlgVyIuBE|tb1jvHLVOA>
# A peek at our input for the transformer
print(xb)


#|%%--%%| <tb1jvHLVOA|REOpA3xQUW>

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token, from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # B - Batch - 4, our batch_size
        # T - Time - 8, our block_size
        # C - Channel - 64, our vocab_size
        # idx - our training input is B by T

        # idx and targets are both (B,T) tensors of integers
        # logits are the predictions based on the identity of iself "what word follows the letter "I" "
        logits = self.token_embedding_table(idx) # (B,T,C)
        
        if targets is None:
            loss = None
        else:
            # evaluation of predictions and targets via log-loss-likelihood
            # cross_entropy only accepts BCT instead of BTC
            B, T, C  = logits.shape #unpack shape into values
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) #BCT format

        return logits, loss

    def generate(self, idx, max_new_tokens):
         # idx is (B, T) arary of indices in the current context
         for _ in range(max_new_tokens):
             # get predictions
             logits, loss = self(idx) #goes to forward function
             # focus only on the last time step
             logits = logits[:, -1, :] # becomes (B, C)
             # apply softmax to get probabilities
             probs = F.softmax(logits, dim=1) # (B, C)
             # sample from multinomial distribution
             idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) because we only want 1 sample
             # append sampled index to the running sequence
             idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb,yb)
print(logits.shape)
print(loss)

# create 1x1 tensor of 0 - 0 cooresponds to a newline character
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long) , max_new_tokens=100)[0].tolist()))

#|%%--%%| <REOpA3xQUW|fKLJZkHK7B>

optimizer = torch.optim.AdamW(m.parameters(),lr=1e-3)

#|%%--%%| <fKLJZkHK7B|5XU95QcTAr>

batch_size = 32
for steps in range(100):

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print(loss.item())


