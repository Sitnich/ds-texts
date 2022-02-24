import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import src.models.model_classes as cl

root_dir = os.path.abspath("..")


def prepare_data(root_dir=root_dir):
    ds_info = pd.read_csv(root_dir + '/data/ds3.csv')
    ds_info = ds_info.dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        ds_info['description'], ds_info['name'], test_size=0.1, random_state=42)

    return X_train, X_test, y_train, y_test


def prepare_dataloader(X, y):
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    dataset = cl.DescDataset(X, y, tokenizer=tokenizer)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return data_loader

def download_model(root_dir = root_dir):
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    torch.save(model.state_dict(), root_dir + '/model/distilgpt2_model.pt')
    torch.save(tokenizer, root_dir + '/model/distilgpt2_token.pt')
