import torch
import torch.nn as nn
import torch.nn.functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get all characters occurred in dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

print(''.join(ch for ch in chars))
print('vocab size: ', vocab_size)

# create a mapping from character to integer
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encode: take a string, output a list of integers
decode = lambda l: ''.join(itos[i] for i in l)  # decode: take a list, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

data = torch.tensor(encode(text), dtype=torch.long)

# split data into train and validation sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

batch_size = 4  # how many sequences will we process in parallel
block_size = 8  # what is the maximum context length of for predictions


def get_batch(split):
    # get a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


xb, yb = get_batch('train')


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(idx)  # (4, 8) -> (4, 8, 65)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)  # idx: (B, T), logits:(B*T, C)
            logits = logits[:, -1, :]
            props = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(props, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel(vocab_size)
m = model.to(device)
logits, loss = m(xb, yb)

# Create a pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for step in range(10000):
    # sample from a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=None)
    loss.backward()
    optimizer.step()

    print(loss.item())

print(decode(m.generate(idx=torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=1000)[0].tolist()))
