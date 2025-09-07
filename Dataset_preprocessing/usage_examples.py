
# ========== USAGE EXAMPLES FOR PROCESSED DATASET ==========

# Example 1: Load data for CNN classification model
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class ProcessedDiabeticDataset(Dataset):
    def __init__(self, data_dir, split='train', technique='contrast_enhanced', transform=None):
        self.data_dir = os.path.join(data_dir, f'{technique}/{split}')
        self.transform = transform
        self.images = [f for f in os.listdir(self.data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Extract label from filename or folder structure
        # Modify this based on your labeling scheme
        label = 0  # Placeholder
        
        return image, label

# Example usage:
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load different processed versions
contrast_dataset = ProcessedDiabeticDataset('/content/processed_dataset', 'train', 'contrast_enhanced', transform)
clahe_dataset = ProcessedDiabeticDataset('/content/processed_dataset', 'train', 'clahe_enhanced', transform)
cropped_dataset = ProcessedDiabeticDataset('/content/processed_dataset', 'train', 'cropped_retina', transform)

# Example 2: Load for vessel segmentation
vessel_dataset = ProcessedDiabeticDataset('/content/processed_dataset', 'train', 'vessel_enhanced', transform)
edge_dataset = ProcessedDiabeticDataset('/content/processed_dataset', 'train', 'edge_detection', transform)

# Example 3: Load augmented data for training
augmented_dataset = ProcessedDiabeticDataset('/content/processed_dataset', 'train', 'augmented', transform)

# Example 4: Combine multiple techniques
from torch.utils.data import ConcatDataset

combined_dataset = ConcatDataset([
    ProcessedDiabeticDataset('/content/processed_dataset', 'train', 'contrast_enhanced', transform),
    ProcessedDiabeticDataset('/content/processed_dataset', 'train', 'clahe_enhanced', transform),
    ProcessedDiabeticDataset('/content/processed_dataset', 'train', 'augmented', transform)
])

# Example 5: Quick data loading function
def load_processed_data(technique='contrast_enhanced', split='train', batch_size=32):
    dataset = ProcessedDiabeticDataset('/content/processed_dataset', split, technique, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'))
    return dataloader

# Load data for training
train_loader = load_processed_data('clahe_enhanced', 'train', 32)
val_loader = load_processed_data('clahe_enhanced', 'val', 32)
test_loader = load_processed_data('clahe_enhanced', 'test', 32)

# Example 6: Technique comparison for model selection
techniques_to_test = ['original', 'contrast_enhanced', 'clahe_enhanced', 'cropped_retina']
results = {}

for technique in techniques_to_test:
    train_data = load_processed_data(technique, 'train')
    val_data = load_processed_data(technique, 'val')
    
    # Train your model here
    # model_accuracy = train_model(model, train_data, val_data)
    # results[technique] = model_accuracy

print("Best technique:", max(results, key=results.get))
