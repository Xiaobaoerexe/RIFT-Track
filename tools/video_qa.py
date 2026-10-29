import json,imageio.v2 as imageio
from PIL import Image,ImageDraw
from pathlib import Path
r=Path('/home/abc/data2/go2_supermimic_lab/reports/ROBOGAUGE_MIGRATION_20261029');out=r/'video_qa';out.mkdir(exist_ok=True);records=[]
for i,p in enumerate(list((r/'videos').rglob('*.mp4'))+list((r/'native_videos').rglob('*.mp4'))):
 reader=imageio.get_reader(p);meta=reader.get_meta_data();n=reader.count_frames();fps=meta['fps'];frames=[]
 for sec in (1,2.5,5,8):frames.append(Image.fromarray(reader.get_data(min(n-1,int(sec*fps)))).resize((480,360)))
 canvas=Image.new('RGB',(960,760),'white');draw=ImageDraw.Draw(canvas)
 for j,img in enumerate(frames):canvas.paste(img,((j%2)*480,30+(j//2)*365))
 draw.text((10,8),str(p.relative_to(r)),fill='black');dst=out/f'{i:02d}.jpg';canvas.save(dst);reader.close()
 records.append({'video':str(p.relative_to(r)),'frames':n,'fps':fps,'duration':n/fps,'contact_sheet':str(dst.relative_to(r))})
(out/'manifest.json').write_text(json.dumps(records,indent=2));print(json.dumps(records))
