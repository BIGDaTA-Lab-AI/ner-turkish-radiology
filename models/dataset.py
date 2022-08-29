import torch
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('pretrained_models/biobert_cased_v1.2')
labels_to_ids = {
    'O': 0,
    'B-ANAT-DP': 1,
    'I-ANAT-DP': 2,
    'B-OBS-DP': 3,
    'I-OBS-DP': 4,
    'B-OBS-DA': 5,
    'I-OBS-DA': 6,
    'B-OBS-U': 7,
    'I-OBS-U': 8,
}
label_all_tokens = True


def align_label(texts, labels):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx
    return label_ids


class DataSequence(torch.utils.data.Dataset):
    def __init__(self, df):
        lb = [i.split() for i in df['labels'].values.tolist()]
        txt = df['sequence'].values.tolist()
        self.texts = [tokenizer(str(i), padding='max_length', max_length=512, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i, j) for i, j in zip(txt, lb)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels