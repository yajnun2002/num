import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import random
class NumEmbedding(nn.Module):
    def __init__(self,numbers,d_model):
        super().__init__()
        self.numbers = numbers
        self.embeddings = nn.ModuleList( [nn.Embedding( number , d_model ) for number in self.numbers ] )
        self.norm = nn.LayerNorm(d_model)
        self.device = "cuda"
    def forward(self,id,batch_size=None):
        if batch_size is None:
            batch_size = id.shape[0]
        start = torch.randint(0,1000000000,size=(batch_size,1),device=self.device)
        id = start + id
        pe_list = [ self.embeddings[i](id % self.numbers[i]) for i in range(len(self.numbers)) ]
        return self.norm(sum(pe_list))

class PositionEmbedding(nn.Module):
    def __init__(self,numbers,d_model):
        super().__init__()
        self.emb = NumEmbedding(numbers=numbers,d_model=d_model)
    def generate(self,batch_size,maxlen):
        return self.emb(  torch.arange(maxlen,device=self.emb.device)[None],batch_size = batch_size  )

class SegmentEmbedding(nn.Module):
    def __init__(self,numbers,d_model,split):
        super().__init__()
        self.emb = NumEmbedding(numbers=numbers,d_model=d_model)
        self.split = split
    def forward(self,x):
        y = (x == self.split).long()
        y = torch.cumsum( y,dim = 1 )
        return self.emb(y)
class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size ,d_model ) 
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        b,t = x.shape
        return self.norm( self.token_emb(x) )

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, drop, maxlen):
        super().__init__()
        assert d_model % nhead == 0
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(drop)
        self.resid_dropout = nn.Dropout(drop)
        rpe = torch.zeros(1, nhead, maxlen, maxlen)
        for i in range(1, maxlen):
            rpe = rpe - torch.tril(torch.ones(maxlen, maxlen), diagonal=-i).view(1, 1, maxlen, maxlen)
        for i in range(nhead):
            rpe[0, i] = rpe[0, i] * 2 **(-8 / nhead * (i + 1))
        self.register_buffer("RPE", rpe)
        self.register_buffer("bias", torch.tril(torch.ones(maxlen, maxlen)).view(1, 1, maxlen, maxlen))
        self.n_head = nhead
        self.n_embd = d_model
        

    def forward(self, x, mask=None):
        B, T, C = x.size()
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + self.RPE[:, :, :T, :T]
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class Mlp(nn.Module):
    def __init__(self,d_model,drop ):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.proj  = nn.Linear(4 * d_model, d_model)
        self.act = NewGELU()
        self.dropout = nn.Dropout(drop)
    def forward(self,x):
        return self.dropout(self.proj(self.act(self.fc(x))))
    
class Block(nn.Module):
    def __init__(self, d_model, nhead, drop, maxlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, nhead=nhead, drop=drop, maxlen=maxlen)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = Mlp(d_model=d_model,drop=drop)

    def forward(self, x , mask=None):
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x) )
        return x
    
class CLIP(nn.Module):
    def __init__(self,n,d_model):
        super().__init__()
        self.n = n
        self.k = nn.Linear(d_model,d_model)
        self.q = nn.Embedding(n,d_model)
    def forward(self,x):
        k = self.k(x)
        q = self.q.weight.T[None]
        att = k @ q * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        #print(att.shape)
        return att

class GPT_block1(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.drop = nn.Dropout(args.drop)
        self.h1 = nn.ModuleList([Block(d_model=args.dmodel, nhead=args.head, drop=args.drop, maxlen=args.maxlen) for _ in range(args.num_layer)])
        self.ln = nn.LayerNorm(args.dmodel)
        self.clips = nn.ModuleList( [CLIP(n=num , d_model=args.dmodel) for num in args.numbers])
        self.clip_fc = nn.Linear( sum(args.numbers)  ,args.dmodel )
    def forward(self,x):
        x = self.drop(x)
        for block in self.h1:
            x = block(x)
        x = self.ln(x)
        clips = [clip(x) for clip in self.clips ]
        clip = torch.concatenate( clips , dim=-1 )
        clip_fc = self.clip_fc(clip)
        return clip_fc
class GPT_block2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.drop2 = nn.Dropout(args.drop)
        self.h2 = nn.ModuleList([Block(d_model=args.dmodel, nhead=args.head, drop=args.drop, maxlen=args.maxlen) for _ in range(args.num_layer)])
        self.ln_f = nn.LayerNorm(args.dmodel)
        self.lm_head = nn.Linear(args.dmodel, args.vocab, bias=True)
        self.clips = nn.ModuleList( [CLIP(n=num , d_model=args.dmodel) for num in args.numbers])
    def forward(self,x):
        x = self.drop2(x)
        for block in self.h2:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
class GPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.token_embedding = Embedding(d_model=args.dmodel, vocab_size=args.vocab)
        self.position_embedding = PositionEmbedding(d_model=args.dmodel,numbers=args.numbers)
        self.segment_embedding = SegmentEmbedding(d_model=args.dmodel,numbers=args.numbers,split = args.split)
        self.block1 = GPT_block1(args=args)
        self.block2 = GPT_block2(args=args)
        self.step = args.step

        #self.pe = args.pe
    def forward(self, x,mask ):
        if self.step == "first":
            B,T = x.shape
            emb = self.token_embedding(x)
            pe = self.position_embedding.generate( batch_size=B , maxlen= T )
            seg_emb = self.segment_embedding(x)

            x = emb + pe + seg_emb
            clip_fc = self.block1(x)
            clip_fc[mask] = 0
            x = emb + pe + seg_emb + clip_fc
            logits = self.block2(x)
            return logits
        
    def generate(self, idx, start):
        b, t = idx.size()
        tmp_start = start + 0
        while True:
            logits = self.forward(idx)
            idx_new = torch.argmax(logits, dim=2)
            idx[torch.arange(b), tmp_start + 1] = idx_new[torch.arange(b), tmp_start]
            if (torch.sum(idx_new[torch.arange(b), tmp_start] != 2) == 0) or (torch.sum(tmp_start == t - 2) != 0):
                break
            tmp_start[idx_new[torch.arange(b), tmp_start] != 2] += 1
        return idx

class GPT_block3(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.drop = nn.Dropout(args.drop)
        self.h = nn.ModuleList([Block(d_model=args.dmodel, nhead=args.head, drop=args.drop, maxlen=args.maxlen) for _ in range(args.num_layer)])
        self.ln = nn.LayerNorm(args.dmodel)
        self.clips = nn.ModuleList( [CLIP(n=num , d_model=args.dmodel) for num in args.numbers])
        self.lm_head = nn.Linear( args.dmodel,args.vocab)
    def forward(self,x):
        x = self.drop(x)
        for block in self.h:
            x = block(x)
        x = self.ln(x)
        clips = [clip(x) for clip in self.clips ]
        clip = torch.concatenate( clips , dim=-1 )
        return clip
class reGPT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.block = GPT_block3(args=args)
        self.mode = args.mode
        self.gpt = GPT(args=args)
        self.re_times = args.re_times
    def forward(self,x,mask):
        if self.mode == "train":
            B,T = x.shape
            token_emb = self.gpt.token_embedding(x)
            pe = self.gpt.position_embedding.generate( batch_size=B , maxlen= T )
            seg_emb = self.gpt.segment_embedding(x)
            emb = token_emb + pe + seg_emb
            
            #b1 ->b3 -> b2
            x = emb + 0
            clip_fc1 = self.gpt.block1(x)
            clip_fc1[mask] = 0

            logits = []
            x = emb + 0
            x[ :,1: ] = x[ :,1: ] + clip_fc1[: , :-1 ]

            for _ in range(self.re_times):
                clip = self.block(x)
                clip_fc = self.gpt.block1.clip_fc(clip)
                clip_fc[mask] = 0
                x = emb + 0
                x[ :,1: ] = x[ :,1: ] + clip_fc[: , :-1 ]

                logits.append( self.gpt.block2(emb + clip_fc) )
            return logits