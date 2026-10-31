from PIL import Image, ImageDraw
from pathlib import Path
p=Path(r'C:\Users\sharo\Documents\Codex\2026-10-31\mak\outputs\cnn-samples')
for name in ('square','circle','cross'):
 im=Image.new('L',(128,128),255); d=ImageDraw.Draw(im)
 if name=='square': d.rectangle((34,34,94,94),fill=0)
 elif name=='circle': d.ellipse((34,34,94,94),fill=0)
 else: d.rectangle((57,25,71,103),fill=0); d.rectangle((25,57,103,71),fill=0)
 im.save(p/f'{name}.png')
