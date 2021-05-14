import transformers

# From : https://github.com/abhi1thakur/bert-sentiment/blob/master/config.py
SEED = 42
DEVICE = "cuda"
MAX_LEN = 331
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-5
N_ACCUMULATE = 1
BERT_PATH = "bert-base-uncased"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

cols_to_remove = ['excerpt', 'id', 'url_legal', 'license', 'standard_error']