from sklearn.datasets import fetch_20newsgroups
from transformers import AutoModel, BertModel, AutoTokenizer, BertTokenizer
from tokenizers.implementations import base_tokenizer


newsgroups_train = fetch_20newsgroups(subset='train')

MODEL_NAME = "bert-base-cased"
encoder = AutoModel.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokens_pt2 = tokenizer(newsgroups_train.data[0], return_tensors="pt", padding=True)
outputs2, pooled2 = encoder(**tokens_pt2)

import torch
embedding = torch.Tensor([])

for j in range(100):
    print("{}/{} finished".format(j, 100))

    encoder = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    for i in range(100):
        tokens_pt2 = tokenizer(newsgroups_train.data[i][:512], return_tensors="pt", padding=True)
        _, pooled2 = encoder(**tokens_pt2)
        embedding = torch.cat([embedding, pooled2])
    print(embedding.shape)

torch.save(embedding, "bert_embedding.pt")
