import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from models.dataset import DataSequence
from models.bert_ner_classifier import BertModel
import numpy as np
from tqdm import tqdm

LEARNING_RATE = 1e-2
EPOCHS = 5

def train_loop(model, df_train, df_val):

    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)

    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label[0].to(device)
            mask = train_data['attention_mask'][0].to(device)
            input_id = train_data['input_ids'][0].to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            logits_clean = logits[0][train_label != -100]
            label_clean = train_label[train_label != -100]

            predictions = logits_clean.argmax(dim=1)

            acc = (predictions == label_clean).float().mean()
            total_acc_train += acc
            total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label[0].to(device)
            mask = val_data['attention_mask'][0].to(device)

            input_id = val_data['input_ids'][0].to(device)

            loss, logits = model(input_id, mask, val_label)

            logits_clean = logits[0][val_label != -100]
            label_clean = val_label[val_label != -100]

            predictions = logits_clean.argmax(dim=1)

            acc = (predictions == label_clean).float().mean()
            total_acc_val += acc
            total_loss_val += loss.item()

        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

        print(
            f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}')


def evaluate(model, df_test):
    test_dataset = DataSequence(df_test)

    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0.0

    for test_data, test_label in test_dataloader:
        test_label = test_label[0].to(device)
        mask = test_data['attention_mask'][0].to(device)
        input_id = test_data['input_ids'][0].to(device)

        loss, logits = model(input_id, mask, test_label.long())

        logits_clean = logits[0][test_label != -100]
        label_clean = test_label[test_label != -100]

        predictions = logits_clean.argmax(dim=1)

        acc = (predictions == label_clean).float().mean()
        print(acc)
        total_acc_test += acc

    val_accuracy = total_acc_test / len(df_test)
    print(f'Test Accuracy: {total_acc_test / len(df_test): .3f}')


if __name__ == "__main__":
    # df = pd.read_csv('data/sequence_labelling/radgraph_train.csv')
    # df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42), [int(.8 * len(df)), int(.9 * len(df))])
    df_train = pd.read_csv('data/sequence_labelling/radgraph_train.csv')
    df_test = pd.read_csv('data/sequence_labelling/radgraph_test.csv')
    df_val = pd.read_csv('data/sequence_labelling/radgraph_dev.csv')

    model = BertModel()

    # train_loop(model, df_train, df_val)
    evaluate(model, df_test)