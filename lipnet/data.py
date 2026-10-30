"""Official VisA one-class split; no category-specific tuning or resplitting."""
from pathlib import Path
import csv
from scripts.run_mvtec_patch_lipnet import TestImage
VISA_CATEGORIES=('candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum')

def visa_split(root):
    root=Path(root).resolve();csv_path=root/'split_csv/1cls.csv'
    with csv_path.open(newline='') as f:rows=list(csv.DictReader(f))
    seen=set();data={c:{'train':[],'test':[]} for c in VISA_CATEGORIES}
    for row in rows:
        c,split,label=row['object'],row['split'],row['label']
        if c not in data or split not in {'train','test'} or label not in {'normal','anomaly'}:
            raise ValueError(f'Unexpected official split entry: {row}')
        image=(root/row['image']).resolve()
        if not image.is_relative_to(root) or not image.is_file() or image in seen:
            raise ValueError(f'Missing/duplicate/overlapping/outside-root image: {image}')
        seen.add(image)
        mask=(root/row['mask']).resolve() if row['mask'] else None
        if mask and (not mask.is_relative_to(root) or not mask.is_file()):raise ValueError(f'Missing/outside-root mask: {mask}')
        if split=='train':
            if label!='normal' or mask is not None:raise ValueError('Training must contain normal images only')
            data[c]['train'].append(image)
        else:
            if (label=='anomaly') != (mask is not None):raise ValueError('Test label/mask mismatch')
            data[c]['test'].append(TestImage(image,int(label=='anomaly'),'anomaly' if label=='anomaly' else 'good',mask))
    if len(rows)!=10821 or sum(len(d['train']) for d in data.values())!=8659:
        raise ValueError('Official VisA 1cls split total differs from the pinned protocol')
    for c,d in data.items():
        if not d['train'] or sum(x.label for x in d['test'])!=100:raise ValueError(f'Incomplete VisA category {c}')
        d['train'].sort();d['test'].sort(key=lambda x:str(x.path))
    return data
