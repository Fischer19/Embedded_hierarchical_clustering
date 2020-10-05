from sklearn.datasets import fetch_20newsgroups
from transformers import AutoModel, BertModel, AutoTokenizer, BertTokenizer
from tokenizers.implementations import base_tokenizer
import argparse
import torch

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--start', required=False, type=int, default=100)
    args = parser.parse_args()

	newsgroups_train = fetch_20newsgroups(subset='train')

	MODEL_NAME = "bert-base-cased"
	encoder = AutoModel.from_pretrained(MODEL_NAME)
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	tokens_pt2 = tokenizer(newsgroups_train.data[0], return_tensors="pt", padding=True)
	outputs2, pooled2 = encoder(**tokens_pt2)

	embedding = torch.Tensor([])

    encoder = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("{}/{} finished".fortmat(args.start, 100))
    for i in range(100):
        tokens_pt2 = tokenizer(newsgroups_train.data[100 * args.start + i][:512], return_tensors="pt", padding=True)
        _, pooled2 = encoder(**tokens_pt2)
        embedding = torch.cat([embedding, pooled2])
    print(embedding.shape)

	torch.save(embedding, "bert_embedding_{}.pt".format(args.start))
