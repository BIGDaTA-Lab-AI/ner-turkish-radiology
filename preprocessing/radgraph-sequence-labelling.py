import json
import csv


if __name__ == "__main__":
    FILENAME = 'MIMIC-CXR_graphs'
    with open('../data/raw/' + FILENAME + '.json') as file:
        data = json.load(file)

    labeled_radgraph = [['sequence', 'labels']]
    for report_name in data:
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
        labeled_radgraph.append([report["text"], ' '.join(sequence_labels)])

    with open('../data/sequence_labelling/' + FILENAME + '.csv', 'w', newline='') as radgraph_csv:
        writer = csv.writer(radgraph_csv, quoting=csv.QUOTE_ALL)
        writer.writerows(labeled_radgraph)

