import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
tokens = { str(i) : i+4 for i in range(11) }
tokens["_"] = 0
tokens["<bos>"] = 1
tokens["<eos>"] = 2
tokens["<ans>"] = 3
for c in "+-*/()=abcdefghijklmnopqrstuvwxyz":
    tokens[c] = len(tokens)

token2char = { value:key for key,value in tokens.items()}
token2char[1] = "<"
token2char[2] = ">"
token2char[3] = ":"
token2char[0] = " "
token2char[tokens["10"]] = "A"
def to_tensor(s):
    return torch.tensor(  [tokens[x] for x in s.split()] ,dtype=int )
def to_sentence(x):
    return "".join([ token2char[int(x[i])] if int(x[i]) in token2char else "?" for i in range(x.size(0))])
import random

class ArithmeticDataset(Dataset):
    def __init__(self,df,maxlen,vocab_size):
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.df = df
        self.X = torch.zeros(len(df) , self.maxlen , dtype=torch.long)
        self.y = torch.zeros(len(df) ,self.maxlen, dtype=torch.long)
        for idx in range(len(df)):
            start = len(self.df["question"][idx].split())
            X = to_tensor(self.df["CoT"][idx])
            y = (X+0)[1:]
            y[:start] = 0
            if len(X) >= maxlen:
                self.X[idx] = X[:maxlen]
            else:
                self.X[idx][ :len(X) ] = X
            
            if len(y) >= maxlen:
                self.y[idx] = y[:maxlen]
            else:
                self.y[idx][:len(y)] = y
    
    def __len__(self):
        return len(self.df)# ...
    def __getitem__(self,idx):
        return self.X[idx],self.y[idx]