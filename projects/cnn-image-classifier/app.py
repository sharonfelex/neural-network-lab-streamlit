from pathlib import Path
import io, numpy as np, torch
from torch import nn
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
app=FastAPI(title="Tiny CNN Classifier",description="Classify simple synthetic shapes with a convolutional neural network.")
app.mount("/static",StaticFiles(directory="static"),name="static")
class Net(nn.Module):
 def __init__(self):
  super().__init__(); self.layers=nn.Sequential(nn.Conv2d(1,8,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(8,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(16*8*8,3))
 def forward(self,x): return self.layers(x)
def make_data(n=600):
 x=torch.zeros(n,1,32,32); y=torch.arange(n)%3
 for i,k in enumerate(y):
  a=int(torch.randint(4,10,())); cx=int(torch.randint(11,21,())); cy=int(torch.randint(11,21,()))
  yy,xx=torch.meshgrid(torch.arange(32),torch.arange(32),indexing='ij')
  if k==0: mask=((xx-cx).abs()<=a)&((yy-cy).abs()<=a)
  elif k==1: mask=(xx-cx)**2+(yy-cy)**2<=a*a
  else: mask=(((yy-cy).abs()<=a)&((xx-cx).abs()<=1))|(((xx-cx).abs()<=a)&((yy-cy).abs()<=1))
  if k==2: mask=(((yy-cy).abs()<=a)&((xx-cx).abs()<=1))|(((xx-cx).abs()<=a)&((yy-cy).abs()<=1))
  x[i,0]=mask.float()+torch.rand(32,32)*.04
 return x,y
@app.get("/")
def home(): return FileResponse("static/index.html")
@app.post("/api/classify")
async def classify(file:UploadFile=File(...)):
 try: img=Image.open(io.BytesIO(await file.read())).convert('L').resize((32,32)); arr=np.asarray(img,dtype=np.float32)/255
 except Exception: return {"error":"Please upload a valid image."}
 x=torch.tensor(arr).reshape(1,1,32,32)
 if x.mean()>.5: x=1-x
 torch.manual_seed(12); net=Net(); X,y=make_data(); opt=torch.optim.Adam(net.parameters(),lr=.004); loss=nn.CrossEntropyLoss()
 for _ in range(100): opt.zero_grad(); l=loss(net(X),y); l.backward(); opt.step()
 with torch.no_grad(): p=net(x).softmax(1)[0].tolist()
 labels=['square','circle','cross']; idx=int(np.argmax(p))
 return {"prediction":labels[idx],"confidence":round(p[idx],3),"scores":{labels[i]:round(p[i],3) for i in range(3)},"note":"The CNN was trained on simple noisy shape examples. Best results: a centered, high-contrast shape."}

