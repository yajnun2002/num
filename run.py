import torch
import torch.optim as optim
import torch.nn as nn
from model import GPT
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_scheduler, set_seed

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from my_token import ArithmeticDataset
class Args:
    def __init__(self):
        self.name = "CLIP"
        self.split = 20
        self.numbers = [3,4,5,7,11]
        self.dmodel = 256
        self.vocab = 64
        self.head = 4
        self.drop = 0.1
        self.maxlen = 120
        self.datalen = 25
        self.batch_size = 100
        self.seed = 514
        self.num_layer = 4
        self.weight_decay  = 0.01
        self.learning_rate = 1e-4 
        self.output_dir = "./output/" + self.name
        self.output_list = []
        self.file_path = "./data/train.csv"
        self.drop = 0.1
        self.warmup = 5
        self.model = GPT
        self.dataset = ArithmeticDataset
        self.epoch = 100
        self.betas = (0.9, 0.999)
        self.model_path = None#"./model/merge/epoch_100.pt"
main_process = 0
args = Args()
os.makedirs( args.output_dir,exist_ok=True )
log_writer = SummaryWriter(log_dir=args.output_dir)
args_pathname = "./args/" + args.name + ".pkl"
import pickle
model = args.model(args).cuda()
if args.model_path:
    model.load_state_dict( torch.load(args.model_path) , strict = True)


import pandas as pd
df = pd.read_csv(args.file_path)
dataset = args.dataset(df = df,maxlen = args.datalen,vocab_size = args.vocab)

log_writer = SummaryWriter(log_dir=args.output_dir)
optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay)

from torch.utils.data.dataloader import DataLoader
loader = DataLoader(dataset , batch_size=args.batch_size,num_workers=0,drop_last=False,shuffle=True)
criterion = nn.CrossEntropyLoss(ignore_index=0)
import random
for epoch in range(args.epoch):
    model.train()
    pbar = tqdm(loader) 
    for data_iter_step, (x, y ) in enumerate(pbar):
        x, y  = x.cuda(), y.long().cuda() 
        mask = y==0
        logits = model(x,mask)
        loss = criterion(logits.transpose(1,2) , y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_1000x = int((data_iter_step / len(loader) + epoch) * 1000)
        log_writer.add_scalar('loss', loss.item(), epoch_1000x)
        pbar.set_description(f"epoch:{epoch}")
    if (epoch+1)%10 == 0:
        filename = f"{args.output_dir}/epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), filename)
        args.output_list.append(filename)
                
        os.makedirs( "args",exist_ok=True )
        with open(args_pathname,"wb") as file:
            pickle.dump( args,file )

    #if (epoch + 1) % (args.epoch // 10) == 0:
    #        torch.save(model.state_dict(), f"{args.output_dir}/epoch_{epoch+1}.pt")
    log_writer.flush()
