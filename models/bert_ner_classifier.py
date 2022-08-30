from transformers import BertForSequenceClassification
from torch import nn

unique_labels = [
    'O',
    'B-ANAT-DP',
    'I-ANAT-DP',
    'B-OBS-DP',
    'I-OBS-DP',
    'B-OBS-DA',
    'I-OBS-DA',
    'B-OBS-U',
    'I-OBS-U'
]


class BertModel(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertModel, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained('pretrained_models/biobert_cased_v1.2', num_labels=len(unique_labels), dropout=dropout)

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output