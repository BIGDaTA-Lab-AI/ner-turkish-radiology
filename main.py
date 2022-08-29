import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from models.dataset import DataSequence
from models.bert_ner_classifier import BertModel
from utils import load_model, save_model
from tqdm import tqdm
import fire

LEARNING_RATE = 5e-3
EPOCHS = 5
BATCH_SIZE = 2


def train(model=BertModel()):
    df_train = pd.read_csv('data/sequence_labelling/radgraph_train.csv')
    df_val = pd.read_csv('data/sequence_labelling/radgraph_dev.csv')
    train_dataset = DataSequence(df_train)
    val_dataset = DataSequence(df_val)
    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=BATCH_SIZE)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    if use_cuda:
        model = model.cuda()
    model.train()

    worst_acc = 0
    worst_loss = 1000
    for epoch_num in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0

        for train_data, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)
            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):
                logits_clean = logits[i][train_label[i] != -100]
                label_clean = train_label[i][train_label[i] != -100]
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
            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)
            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):
                logits_clean = logits[i][val_label[i] != -100]
                label_clean = val_label[i][val_label[i] != -100]
                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                total_acc_val += acc
                total_loss_val += loss.item()

        train_loss = total_loss_train / len(df_train)
        train_accuracy = total_acc_train / len(df_train)
        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

        if val_loss < worst_loss:
            save_model(model, epoch_num)

        print(f'Epoch: {epoch_num} | Train_Loss: {train_loss: .3f} | Train_Accuracy: {train_accuracy: .3f} | Val_Loss: {val_loss: .3f} | Val_Accuracy: {val_accuracy: .3f}')


def evaluate(model=BertModel()):
    df_test = pd.read_csv('data/sequence_labelling/radgraph_test.csv')
    test_dataset = DataSequence(df_test)
    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1)
    model = load_model(model)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
    model.eval()

    total_acc_test = 0.0
    for test_data, test_label in test_dataloader:
        test_label = test_label.to(device)
        mask = test_data['attention_mask'].squeeze(1).to(device)
        input_id = test_data['input_ids'].squeeze(1).to(device)
        loss, logits = model(input_id, mask, test_label)

        for i in range(logits.shape[0]):
            logits_clean = logits[i][test_label[i] != -100]
            label_clean = test_label[i][test_label[i] != -100]
            predictions = logits_clean.argmax(dim=1)
            acc = (predictions == label_clean).float().mean()
            total_acc_test += acc

    val_accuracy = total_acc_test / len(df_test)
    print(f'Test Accuracy: {val_accuracy: .3f}')


if __name__ == "__main__":
    fire.Fire()