import random, json
from pycocotools.coco import COCO

def generator(dataset, index):
    """
    Prompt generator by COCO's captions.
    """
    
    parts = []
    for _ in range(1):
        id = random.choice(index)
        annotations = dataset.loadAnns(dataset.getAnnIds(imgIds=id))
        part = random.choice(annotations)
        parts.append(part['caption'].rstrip('. ').strip())
    return part['caption'].lower()

def save_prompts():
    """
    Save prompts generated in a json file.
    """
    
    prompts = set()
    dataset = COCO("data/util/annotations/captions_val2017.json")
    index = dataset.getImgIds()
    
    for _ in range(5000):
        prompt = generator(dataset, index)
        prompts.add(prompt)
    
    with open("./data/util/prompts.json", "w", encoding="utf-8") as file:
        json.dump(list(prompts), file, indent=1)
    return

def load_prompts():
    """
    Prompts loader from json file.
    """
    
    file_path = "./data/util/prompts.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)  
    return data