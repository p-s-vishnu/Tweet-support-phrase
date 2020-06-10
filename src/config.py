## TODO: What does tokenizers do
# tokernizers by huggingface
import tokenizers
import os

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16
EPOCHS = 5
BERT_PATH = "../bert_base_uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/tweet-sentiment-extraction/train.csv"
os.curdir
TOKENIZER = tokenizers.BertWordPieceTokenizer(
    os.path.join(BERT_PATH, "vocab.txt"),
    lowercase=True
)
