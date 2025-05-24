import random, json
from pycocotools.coco import COCO

def generator(dataset, index):
    parts = []
    for _ in range(2):
        id = random.choice(index)
        annotations = dataset.loadAnns(dataset.getAnnIds(imgIds=id))
        part = random.choice(annotations)
        parts.append(part['caption'].rstrip('. ').strip())
    return " and ".join(parts)

def save_prompts():
    prompts = set()
    dataset = COCO("data/raw/annotations/captions_val2017.json")
    index = dataset.getImgIds()
    
    for _ in range(5000):
        prompt = generator(dataset, index).lower()
        prompts.add(prompt)
    
    with open("./data/raw/prompts.json", "w", encoding="utf-8") as file:
        json.dump(list(prompts), file, indent=1)
    return
        
def load_prompts():
    file_path = "./data/raw/prompts.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    return data