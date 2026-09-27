"""23 (fast) — C4-equivariant G-CNN + rotation-as-group-task. Downsampled to 14x14."""
import sys, math
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/lain/paradox-logic")
from mnist_loader import load_mnist
torch.manual_seed(0); G = 4

class Lift(nn.Module):
    def __init__(s, C): super().__init__(); s.conv = nn.Conv2d(1, C, 3, padding=1)
    def forward(s, x):
        B = x.shape[0]
        xs = torch.stack([torch.rot90(x, k, (2, 3)) for k in range(G)], 1)
        y = s.conv(xs.reshape(B*G, 1, *x.shape[2:]))
        return y.reshape(B, G, -1, *x.shape[2:]).permute(0, 2, 1, 3, 4)

class GConv(nn.Module):
    def __init__(s, Ci, Co, k=3):
        super().__init__()
        s.W = nn.Parameter(torch.randn(G, Co, Ci, k, k)*(1/math.sqrt(Ci*k*k)))
    def forward(s, x):
        B, Ci, _, H, W = x.shape; y = 0
        for t in range(G):
            xs = torch.roll(x, -t, 2).permute(0,2,1,3,4).reshape(B*G, Ci, H, W)
            c = F.conv2d(xs, s.W[t], padding=1).reshape(B,G,-1,H,W).permute(0,2,1,3,4)
            y = c if y is 0 else y+c
        return y

def gpool(x):
    B,C,Gg,H,W = x.shape
    return F.max_pool2d(x.reshape(B*C*Gg,1,H,W),2).reshape(B,C,Gg,H//2,W//2).mean((3,4))

class GCNN(nn.Module):
    def __init__(s, C1=8, C2=16):
        super().__init__(); s.lift=Lift(C1); s.g1=GConv(C1,C2); s.g2=GConv(C2,C2)
        s.fc_d=nn.Linear(C2,10); s.fc_r=nn.Linear(C2*G,G)
    def forward(s, x):
        z = F.max_pool2d(x, 2)                      # 28->14
        z = F.relu(s.lift(z)); z = F.relu(s.g1(z)); z = F.relu(s.g2(z))
        z = gpool(z)                                # (B,C2,G)
        return s.fc_d(z.mean(2)), s.fc_r(z.flatten(1))

class CNN(nn.Module):
    def __init__(s):
        super().__init__(); s.c1=nn.Conv2d(1,16,3,padding=1); s.c2=nn.Conv2d(16,32,3,padding=1); s.fc=nn.Linear(32*3*3,10)
    def forward(s, x):
        x=F.max_pool2d(F.relu(s.c1(F.max_pool2d(x,2))),2)
        x=F.max_pool2d(F.relu(s.c2(x)),2)
        return s.fc(x.flatten(1))

Xt,Yt,Xe,Ye = load_mnist("/tmp/opencode/mnist")
Xtr=torch.tensor(Xt[:8000]).float().reshape(-1,1,28,28)/255.; Ytr=torch.tensor(Yt[:8000]).long()
Xte=torch.tensor(Xe[:2000]).float().reshape(-1,1,28,28)/255.; Yte=torch.tensor(Ye[:2000]).long()
rot=lambda x,k: torch.rot90(x,k,(2,3))

print("(A) digit classification, train UPRIGHT only")
cnn=CNN(); opt=torch.optim.Adam(cnn.parameters(),1e-3)
for ep in range(4):
    p=torch.randperm(len(Xtr))
    for s in range(0,len(Xtr),128):
        i=p[s:s+128]; opt.zero_grad(); F.cross_entropy(cnn(Xtr[i]),Ytr[i]).backward(); opt.step()
gcn=GCNN(); opt=torch.optim.Adam(gcn.parameters(),3e-3)
for ep in range(9):
    p=torch.randperm(len(Xtr))
    for s in range(0,len(Xtr),128):
        i=p[s:s+128]; opt.zero_grad(); d,_=gcn(Xtr[i]); F.cross_entropy(d,Ytr[i]).backward(); opt.step()
with torch.no_grad():
    ac=[(cnn(rot(Xte,k)).argmax(1)==Yte).float().mean().item() for k in range(4)]
    ag=[(gcn(rot(Xte,k))[0].argmax(1)==Yte).float().mean().item() for k in range(4)]
print(f"    CNN     ({sum(p.numel() for p in cnn.parameters())}p): "+"  ".join(f"{v:.3f}" for v in ac))
print(f"    C4-GCNN ({sum(p.numel() for p in gcn.parameters())}p): "+"  ".join(f"{v:.3f}" for v in ag))

print("(B) rotation (C4 element) classification = group task")
Xs=torch.cat([rot(Xtr,k) for k in range(4)]); ks=torch.tensor(sum([[k]*len(Xtr) for k in range(4)],[]))
g2=GCNN(); opt=torch.optim.Adam(g2.parameters(),3e-3)
for ep in range(4):
    p=torch.randperm(len(Xs))
    for s in range(0,len(Xs),128):
        i=p[s:s+128]; opt.zero_grad(); _,r=g2(Xs[i]); F.cross_entropy(r,ks[i]).backward(); opt.step()
with torch.no_grad():
    ar=[(g2(rot(Xte,k))[1].argmax(1)==k).float().mean().item() for k in range(4)]
print("    rotation acc @ 0/90/180/270: "+"  ".join(f"{v:.3f}" for v in ar))
