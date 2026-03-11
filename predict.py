import torch
from model import CharRNN


languages = ["nno", "nob"]

model = torch.load("model.pt", weights_only=False)


def build_vocab(text):
    return sorted(set(text))



while True:
    text = input("tekst: ")

    if len(text) < 20:
        print("Du må skrive minst 20 bokstavar i setninga :)")
        continue

    chars = sorted(set(text))
    n_chars = len(chars)


    tensor = torch.zeros(len(text), 1, c_chars)

    for i, c in enumerate(text):
        tensor[i][0][chars.index(c)] = 1


    out = model(tensor)

    _, idx = out.topk(1)

    print("Eg antek at dette er:", languages[idx])
     
