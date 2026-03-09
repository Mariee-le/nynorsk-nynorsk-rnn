import torch 
import random

from dataset_loader import load_nb_dataset
from model import CharRNN


texts, labels = load_nb_dataset(limit=8000, min_len=20)

languages = ["nno", "nob"]

all_chars = sorted(set("".join(texts)))
n_chars = len(all_chars)


def char_index(c):
   return all_chars.index(c)


def text_tensor(text):

    tensor = torch.zeros(len(text), 1, n_chars)

    for i, c in enumerate(text):

        if c in all_chars:
            tensor[i][0][char_index(c)] = 1

    return tensor


model = CharRNN(n_chars, 32, len(languages))

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


for step in range(3000):
    
    i = random.randrange(len(texts))

    text_sample = texts[i]
    lang = labels[i]

    x = text_tensor(text_sample)
    y = torch.tensor([languages.index(lang)])

    out = model(x)

    loss = criterion(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(step, loss.item(), lang, text[:50])


torch.save(model, "model.pt")
