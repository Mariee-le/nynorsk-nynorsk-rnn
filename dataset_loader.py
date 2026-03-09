from datasets import load_dataset


def load_nb_dataset(limit=10000, min_len=20):
    
    dataset = load_dataset("NbAiLab/nbnn_language_detection")

    texts = []
    labels = []

    for item in dataset["train"]:

        text = item["text"]
        lang = item["language"]

        # velg kun nynorsk og bokmål
        if lang not in ["nno", "nob"]:
            continue

        # velg antal bokstavar
        if len(text) < min_len:
            continue

        texts.append(text)
        labels.append(lang)
 
        if len(texts) >= limit:
            break

    return texts, labels
