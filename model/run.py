import torch
import torch.optim as optim
import torch.nn as nn
from model import GPT
from model import reGPT
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import get_scheduler, set_seed

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from my_token import ArithmeticDataset
class Args:
    def __init__(self):
        self.name = "reGPT3"
        self.step = "first"
        self.split = 20
        self.pe = False
        self.numbers = [3,4,5,7,11]
        self.dmodel = 256
        self.vocab = 64
        self.head = 4
        self.drop = 0.1
        self.maxlen = 120
        self.batch_size = 50
        self.seed = 514
        self.num_layer = 2
        self.weight_decay  = 0.01
        self.learning_rate = 1e-4 
        self.output_dir = "./output/" + self.name
        self.output_list = []
        self.file_path = "./data/train.csv"
        self.drop = 0.1
        self.warmup = 5
        self.model = reGPT
        self.dataset = ArithmeticDataset
        self.epoch = 30
        self.betas = (0.9, 0.999)
        self.mode = "train"
        self.model_path = "./model/reGPT.pt" #"./model/merge/epoch_100.pt"
        self.re_times = 4
main_process = 0
args = Args()
os.makedirs( args.output_dir,exist_ok=True )
log_writer = SummaryWriter(log_dir=args.output_dir)
args_pathname = "./args/" + args.name + ".pkl"
import pickle
model = args.model(args)
model.load_state_dict(torch.load(args.model_path), strict=True)
model = model.cuda()

import pandas as pd
df = pd.read_csv(args.file_path)
dataset = args.dataset(df = df,maxlen = args.maxlen,vocab_size = args.vocab)
for i in range(args.re_times):
    os.makedirs( args.output_dir+ "/" + str(i),exist_ok=True )
log_writer = [SummaryWriter(log_dir=args.output_dir + "/" + str(_)) for _ in range(args.re_times)]
optimizer = optim.AdamW(model.block.parameters(), lr=args.learning_rate, betas=args.betas, weight_decay=args.weight_decay)

from torch.utils.data.dataloader import DataLoader
loader = DataLoader(dataset , batch_size=args.batch_size,num_workers=0,drop_last=False,shuffle=True)
criterion = nn.CrossEntropyLoss(ignore_index=0)
import random
for epoch in range(args.epoch):
    model.train()
    pbar = tqdm(loader) 
    for data_iter_step, (x, y ) in enumerate(pbar):
        x, y  = x.cuda(), y.long().cuda() 
        mask = (y==0)
        logits = model(x,mask)
        loss_list = []
        for i in range(args.re_times):
            loss_list.append(criterion(logits[i].transpose(1,2) , y))
        loss = sum(loss_list)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_1000x = int((data_iter_step / len(loader) + epoch) * 1000)
        for i in range(args.re_times):
            log_writer[i].add_scalar('loss', loss_list[i].item(), epoch_1000x)
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
    for i in range(args.re_times):
        log_writer[i].flush()
