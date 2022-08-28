import json
import csv


if __name__ == "__main__":
    FILENAME = 'chexpert'
    with open('data/raw/' + FILENAME + '.json') as file:
        data = json.load(file)

    labeled_radgraph = [['label', 'token']]
    for report_name in data:
        report = data[report_name]
        tokens = report["text"].split(' ')
        labeled_tokens = [['O', token] for token in tokens]
        if "entities" in report:
            entities = report["entities"]
        else:
            entities = report["labeler_1"]["entities"]

        for i in entities:
            entity = report["entities"][i]
            labeled_tokens[entity["start_ix"]][0] = 'B-' + entity["label"]
            if entity["end_ix"] > entity["start_ix"]:
                for j in range(entity["start_ix"]+1, entity["end_ix"]+1):
                    labeled_tokens[j][0] = 'I-' + entity["label"]
        print(labeled_tokens)
        labeled_radgraph = labeled_radgraph + labeled_tokens

    with open('data/' + FILENAME + '.csv', 'w', newline='') as radgraph_csv:
        writer = csv.writer(radgraph_csv, quoting=csv.QUOTE_ALL)
        writer.writerows(labeled_radgraph)