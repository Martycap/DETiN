import torch, cv2
from torchvision import transforms
from torch.utils.data import Dataset

class CNNDataset(Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform if transform else transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        original_path, tampered_path, mask_path = self.triplets[idx]
        
        original = cv2.imread(original_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        tampered = cv2.imread(tampered_path)
        tampered = cv2.cvtColor(tampered, cv2.COLOR_BGR2RGB)

        original = self.transform(original)  
        tampered = self.transform(tampered)  

        image_pair = torch.cat((original, tampered), dim=0)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (256, 256))
        mask = torch.tensor(mask / 255., dtype=torch.float32).unsqueeze(0)  

        return image_pair, mask

