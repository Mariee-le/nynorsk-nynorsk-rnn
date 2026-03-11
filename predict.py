import torch
from model import CharRNN


languages = ["nno", "nob"]

model = torch.load("model.pt", weights_only=False)

all_chars = sorted(set("".join([t for t,_ in zip(range(8000), range(8000))])))

model.eval()

def char_index(c):
    return all_chars.index(c)

def text_tensor(text):
    tensor = torch.zeros(len(text), 1, len(all_chars))
    for i, c in enumerate(text):
        if c in all_chars:
            tensor[i][0][char_index(c)] = 1
    return tensor



while True:
    text = input("tekst: ")

    if len(text) < 20:
        print("Du må skrive minst 20 bokstavar i setninga :)")
        continue
        
    x = text_tensor(text)
    out = model(x)
    _, idx = out.topk(1)
    print("Eg antek at dette er:", languages[idx.item()])
     
