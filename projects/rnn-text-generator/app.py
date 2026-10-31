from pathlib import Path
import torch
from torch import nn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
app=FastAPI(title="Tiny RNN Text Generator",description="A character-level recurrent neural network trained on a tiny sample corpus.")
app.mount("/static",StaticFiles(directory="static"),name="static")
CORPUS=("a recurrent neural network reads a sequence one character at a time. " "it keeps a hidden state that carries context forward. " "small models can learn patterns in text and generate new sequences. ")*12
class RNN(nn.Module):
 def __init__(self,n):
  super().__init__(); self.embed=nn.Embedding(n,24); self.rnn=nn.GRU(24,48,batch_first=True); self.out=nn.Linear(48,n)
 def forward(self,x): y,_=self.rnn(self.embed(x)); return self.out(y)
class Request(BaseModel): seed:str="the "; length:int=160
@app.get("/")
def home(): return FileResponse("static/index.html")
@app.post("/api/generate")
def generate(req:Request):
 chars=sorted(set(CORPUS)); ids={c:i for i,c in enumerate(chars)}; torch.manual_seed(4); model=RNN(len(chars)); opt=torch.optim.Adam(model.parameters(),lr=.012); loss=nn.CrossEntropyLoss(); seq=torch.tensor([ids[c] for c in CORPUS])
 for _ in range(100):
  starts=torch.randint(0,len(seq)-41,(64,)); batch=torch.stack([seq[s:s+40] for s in starts]); target=torch.stack([seq[s+1:s+41] for s in starts]); opt.zero_grad(); l=loss(model(batch).reshape(-1,len(chars)),target.reshape(-1)); l.backward(); opt.step()
 seed=''.join(c for c in req.seed.lower() if c in ids) or 'a '; out=seed; inp=torch.tensor([[ids[c] for c in seed]])
 with torch.no_grad():
  for _ in range(max(1,min(req.length,400))):
   logits=model(inp)[0,-1]; probs=torch.softmax(logits/.8,0); nxt=int(torch.multinomial(probs,1)); out+=chars[nxt]; inp=torch.tensor([[ids[c] for c in out]])
 return {"text":out,"note":"A small character-level GRU trained on a short example corpus. Expect playful, imperfect text."}

