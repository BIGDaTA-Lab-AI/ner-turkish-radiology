# ner-turkish-radiology

## Download Data
- Visit https://physionet.org/content/radgraph/1.0.0
- Login from the menu bar to get access to Files section
- Download the following files:
  1. *CheXpert_graphs.json*,
  2. *MIMIC-CXR_graphs.json*,
  3. *dev.json*,
  4. *test.json*, and
  5. *train.json*
- Put the downloaded files in *data/raw* directory

## Data Preprocessing
- Generate training dataset: `python utils.py json_to_csv --input_files="['chexpert','MIMIC-CXR_graphs']" --output_file="train" --no_of_samples=4000`
- Generate validation dataset: `python utils.py json_to_csv --input_files="['test', 'dev']" --output_file="dev"`
- Generate test dataset: `python utils.py json_to_csv --input_files="['train']" --output_file="test"`

## Download Pretrained Models
### Biobert Base Cased v1.2
- `curl --remote-name-all https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/resolve/main/config.json https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/resolve/main/pytorch_model.bin https://huggingface.co/dmis-lab/biobert-base-cased-v1.2/resolve/main/vocab.txt --output-dir "pretrained_models/biobert_cased_v1.2"`

## Virtual Environment and Requirements Installation
- If no virtual environment (venv), create it: `python3 -m venv venv`
- Activate virtual environment: `source venv/bin/activate`
- Install requirements: `pip install -r requirements.txt`

## Run
- Train: `python main.py train`
- Evaluate: `python main.py evaluate`
- Evaluate on text: `python main.py evaluate_one_text --sentence="Increased right lower lobe capacity, concerning for infection"`
