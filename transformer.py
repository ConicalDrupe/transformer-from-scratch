
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
print(data[:500]) # first 500 characters again

