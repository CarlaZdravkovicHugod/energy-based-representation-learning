
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import math
import argparse
from utils.neptune_logger import NeptuneLogger

# optional deps ---------------------------------------------------------------
try:
    from compressai.layers import GDN, IGDN  # Ballé et al. 2018
except ImportError:
    class _Id(nn.Identity):
        def __init__(self,*a,**k): super().__init__()
    GDN=IGDN=_Id
try:
    from piqa import MSSSIM                     # Wang et al. 2003 (MS‑SSIM)
except ImportError:
    MSSSIM=None
try:
    import lpips                                # Zhang et al. 2018 (LPIPS)
except ImportError:
    lpips=None


class ConvGN(nn.Sequential):
    """Conv ▸ GroupNorm ▸ GELU."""
    def __init__(self,in_c,out_c,k=3,s=1,p=1,groups=8):
        super().__init__(nn.Conv2d(in_c,out_c,k,s,p,bias=False),
                         nn.GroupNorm(groups,out_c),
                         nn.GELU())

class ResAttnGDN(nn.Module):
    """Residual block with MH-Attention + GDN (Ballé 2018)."""
    def __init__(self,ch,heads=4):
        super().__init__()
        self.conv1=ConvGN(ch,ch)
        self.conv2=ConvGN(ch,ch)
        self.attn=nn.MultiheadAttention(ch,heads,batch_first=True)
        self.gdn=GDN(ch)

    def forward(self,x):
        y=self.conv1(x);y=self.conv2(y)
        b,c,h,w=y.shape
        y=y.flatten(2).transpose(1,2)
        y,_=self.attn(y,y,y)
        y=y.transpose(1,2).view(b,c,h,w)
        y=self.gdn(y)
        return x+0.1*y

# ---------------- Mask/patch helpers (for MAE) --------------------------------
class PatchEmbed(nn.Module):
    """2-D patchify + linear projection (ViT style, Dosovitskiy et al. 2020)."""
    def __init__(self,patch=16,in_c=1,embed=768):
        super().__init__()
        self.patch=patch
        self.proj=nn.Conv2d(in_c,embed,kernel_size=patch,stride=patch)
    def forward(self,x):
        # B,C,H,W -> B,N,embed where N = (H*W)/patch²
        x=self.proj(x)
        x=x.flatten(2).transpose(1,2)
        return x

# MAE backbone (He et al. 2021) - ViT encoder, conv decoder
class MAEDecoder(nn.Module):
    def __init__(self,embed,patch,in_c=1):
        super().__init__();self.patch=patch
        self.proj=nn.Linear(embed,patch*patch*in_c)

    def forward(self,tok,B,H,W):
        x=self.proj(tok)  # B*N,patch²*C
        x=x.view(B,H//self.patch,W//self.patch,self.patch,self.patch,1)
        x=x.permute(0,5,1,3,2,4).contiguous().view(B,1,H,W)
        return x

class MAETop(nn.Module):
    def __init__(self,patch=16,embed=768,depth=6,heads=8,mask_ratio=0.75,in_c=1):
        super().__init__();self.patch=patch;self.mask_ratio=mask_ratio
        self.patchify=PatchEmbed(patch,in_c,embed)
        self.pos_emb=nn.Parameter(torch.zeros(1,1024,embed)) # 1024 patches max
        encoder_layer=nn.TransformerEncoderLayer(embed,heads,embed*4,act="gelu",batch_first=True)
        self.encoder=nn.TransformerEncoder(encoder_layer,depth)
        self.mask_tok=nn.Parameter(torch.zeros(1,1,embed))
        self.decoder=MAEDecoder(embed,patch,in_c)
        nn.init.trunc_normal_(self.pos_emb,std=.02);nn.init.trunc_normal_(self.mask_tok,std=.02)
    def _rand_mask(self,N,device):
        ids=torch.randperm(N,device=device);n_mask=int(N*self.mask_ratio)
        return ids[n_mask:],ids[:n_mask]  # keep, mask
    def forward(self,x):
        B,C,H,W=x.shape;patch_N=(H//self.patch)*(W//self.patch)
        tok=self.patchify(x)+self.pos_emb[:,:patch_N]
        keep,mask=self._rand_mask(patch_N,x.device)
        x_vis=tok[:,keep]
        enc=self.encoder(x_vis)
        # re‑insert mask tokens
        dec_tok=torch.zeros(B,patch_N,enc.size(-1),device=x.device)
        dec_tok[:,keep]=enc;dec_tok[:,mask]=self.mask_tok
        rec=self.decoder(dec_tok.view(B*patch_N,-1),B,H,W)
        return rec,torch.tensor(0.0,device=x.device)  # no VQ loss

###############################################################################
# 4. Dataset, losses, train/val loops, CLI
###############################################################################
class NumpyMRIDataset(Dataset):
    def __init__(self,root:Path,split:str,size:int|None):
        root=Path(root);paths=sorted(root.glob("*.npy"));cut=int(0.8*len(paths))
        self.paths=paths[:cut] if split=="train" else paths[cut:];tfms=[]
        if size:tfms.append(T.Resize((size,size),antialias=True))
        self.tf=T.Compose(tfms)
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        x=np.load(self.paths[i]).astype(np.float32);x=(x-x.min())/(x.max()-x.min()+1e-6)
        if x.ndim==2:x=x[None]
        x=torch.from_numpy(x);x=self.tf(x)
        return x

class MixedLoss(nn.Module):
    def __init__(self,mse=0.8,ms=0.15,lp=0.05):
        super().__init__();self.mse=mse;self.ms=ms if MSSSIM else 0.0;self.lp=lp if lpips else 0.0
        if MSSSIM:self.msfn=MSSSIM(n_channels=1);self.msfn.to("cuda" if torch.cuda.is_available() else "cpu")
        if lpips:self.lpfn=lpips.LPIPS(net="vgg").eval()
    def forward(self,x,y):
        loss=self.mse*F.mse_loss(x,y)
        if self.ms:loss+=self.ms*(1-self.msfn(x,y))
        if self.lp:loss+=self.lp*self.lpfn(x.repeat(1,3,1,1),y.repeat(1,3,1,1)).mean()
        return loss

def psnr_from_mse(mse):return 20*math.log10(1.0/math.sqrt(mse+1e-12))

def save_grid(x,rec,path,n=8):grid=make_grid(torch.cat([x[:n],rec[:n]],0),nrow=n);save_image(grid,path)

def train_epoch(model,loader,crit,opt,scaler,device,beta):
    model.train();tot=0
    for x in loader:
        x=x.to(device)
        with torch.cuda.amp.autocast():
            rec,aux=model(x);loss=crit(rec,x)+beta*aux
        scaler.scale(loss).backward();scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        tot+=loss.item()*x.size(0)
    return tot/len(loader.dataset)

def eval_epoch(model,loader,device):
    model.eval();mse=0
    with torch.no_grad():
        for x in loader:
            x=x.to(device);rec,_=model(x);mse+=F.mse_loss(rec,x,reduction="sum").item()
    mse/=len(loader.dataset);return psnr_from_mse(mse)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train_dir",required=True)
    ap.add_argument("--epochs",type=int,default=200)
    ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--size",type=int,default=256)
    ap.add_argument("--out",default="runs/modular")
    ap.add_argument("--description",default="MAE")
    args=ap.parse_args()
    neptune_logger = NeptuneLogger(False, description=args.description)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds=NumpyMRIDataset(args.train_dir,"train",args.size);val_ds=NumpyMRIDataset(args.train_dir,"val",args.size)
    tl=DataLoader(train_ds,args.batch,shuffle=True,num_workers=4,pin_memory=True)
    vl=DataLoader(val_ds,args.batch,False,num_workers=4,pin_memory=True)
    model=MAETop().to(device)
    beta=0.0
    ema=AveragedModel(model)
    opt=optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4);sched=CosineAnnealingLR(opt,args.epochs)
    crit=MixedLoss().to(device);scaler=torch.cuda.amp.GradScaler()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    best=-1e9
    for ep in range(1,args.epochs+1):
        tloss=train_epoch(model,tl,crit,opt,scaler,device,beta);sched.step();ema.update_parameters(model)
        psnr=eval_epoch(ema,vl,device)
        print(f"Ep{ep:03d} train-loss {tloss:.4f}  PSNR {psnr:.2f}dB")
        neptune_logger.log_metric("train_loss", tloss)
        neptune_logger.log_metric("eval_psnr", psnr)
        neptune_logger.log_metric("epoch", ep)
        neptune_logger.log_metric("lr", opt.param_groups[0]["lr"])
        if psnr>best:
            best=psnr;torch.save({"ep":ep,"state":ema.module.state_dict()},str(out/"best.pt"))
            save_grid(next(iter(vl)).to(device),ema(next(iter(vl)).to(device))[0],str(out/"best_grid.png"))
            neptune_logger.log_model(str(out/"best.pt"), "best.pt")
    print("Best PSNR",best)

if __name__=="__main__":main()
