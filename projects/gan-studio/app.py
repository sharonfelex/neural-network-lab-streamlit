from pathlib import Path
import torch
from torch import nn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app=FastAPI(title="Tiny GAN Studio", description="A tiny GAN learns a one-dimensional bell curve.")
app.mount("/static",StaticFiles(directory="static"),name="static")
class Generator(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(1,16),nn.ReLU(),nn.Linear(16,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x): return self.net(x)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(1,16),nn.LeakyReLU(.2),nn.Linear(16,1),nn.Sigmoid())
    def forward(self,x): return self.net(x)
class Request(BaseModel): count:int=200
@app.get("/")
def home(): return FileResponse("static/index.html")
@app.post("/api/generate")
def generate(req:Request):
    torch.manual_seed(7); G=Generator(); D=Discriminator(); loss=nn.BCELoss(); go=torch.optim.Adam(G.parameters(),lr=.003); do=torch.optim.Adam(D.parameters(),lr=.003)
    for _ in range(350):
        real=torch.randn(96,1); noise=torch.randn(96,1); fake=G(noise).detach(); do.zero_grad(); ld=loss(D(real),torch.ones(96,1))+loss(D(fake),torch.zeros(96,1)); ld.backward(); do.step()
        go.zero_grad(); fake=G(torch.randn(96,1)); lg=loss(D(fake),torch.ones(96,1)); lg.backward(); go.step()
    with torch.no_grad(): vals=G(torch.randn(max(1,min(req.count,500)),1)).flatten().tolist()
    return {"samples":[round(x,3) for x in vals],"note":"Trained a small generator against samples from a normal distribution."}
