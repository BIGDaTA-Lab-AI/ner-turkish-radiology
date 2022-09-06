import torch
import os
import json
import random
import csv
import fire


def save_model(model, epoch, is_best_epoch, directory='model_checkpoint'):
    if not os.path.exists(directory):
        os.mkdir(directory)

    checkpoint_name = 'epoch_{}'.format(epoch)
    path = os.path.join(directory, checkpoint_name)
    torch.save(model.state_dict(), path)
    print('Saved model at epoch {} successfully'.format(epoch))
    if is_best_epoch:
        with open('{}/checkpoint'.format(directory), 'w') as file:
            file.write(checkpoint_name)
            print('Write to checkpoint')


def load_model(model, checkpoint_name=None, directory='model_checkpoint'):
    if checkpoint_name is None:
        with open('{}/checkpoint'.format(directory)) as file:
            content = file.read().strip()
            path = os.path.join(directory, content)
    else:
        path = os.path.join(directory, checkpoint_name)

    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(path))
    return model


def json_to_csv(input_files=["radgraph_train"], output_file="test", no_of_samples=4000):
    data = {}
    for file_name in input_files:
        with open('data/raw/' + file_name + '.json') as file:
            data = data | json.load(file)

    labeled_csv = [['sequence', 'labels']]
    report_names = list(data.keys())
    random.shuffle(report_names)
    count = 0
    for report_name in report_names:
        report = data[report_name]
        sequence_labels = []
        if "entities" in report:
            entities = report["entities"]
        else:
            entities = report["labeler_1"]["entities"]

        for i in entities:
            entity = entities[i]
            labeled_count = len(sequence_labels)
            if labeled_count < entity["start_ix"]:
                sequence_labels = sequence_labels + ['O' for i in range(labeled_count, entity["start_ix"])]
            sequence_labels.append('B-' + entity["label"])
            if entity["end_ix"] > entity["start_ix"]:
                sequence_labels = sequence_labels + ['I-' + entity["label"] for i in range(entity["start_ix"], entity["end_ix"])]
        tokens_count = len(report["text"].split(' '))
        sequence_labels = sequence_labels + ['O' for i in range(len(sequence_labels), tokens_count)]
        labeled_csv.append([report["text"], ' '.join(sequence_labels)])
        count += 1
        if count == no_of_samples:
            break

    with open('data/' + output_file + '.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        writer.writerows(labeled_csv)
    print(str(count) + ' samples converted')


if __name__ == "__main__":
    fire.Fire()