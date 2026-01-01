"""
TRAINING SCRIPT FOR OPEN IMAGES DATASET (OID v7) - XML VERSION
================================================================
This script trains both detection and classification models on your downloaded OID data.
Uses Pascal VOC XML annotations instead of YOLO format.

Usage:
1. Download data: oid_v7_download --classes Person Dog Cat Horse Elephant Bear --type_data train validation --limit 5000 --dest datasets/part_a/
2. Run: python train_models.py
3. Models will be saved to models/ directory
4. Use trained models with main.py for inference

Directory Structure Expected:
datasets/part_a/
    train/
        Person/
            images/
                person_001.jpg
            pascal/
                person_001.xml  (Pascal VOC format)
        Dog/
            images/
            pascal/
        ...
    validation/
        (same structure)

Author: Training Implementation for OID Dataset with XML
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import resnet50
import cv2
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image
import wandb
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')



# ============================================================================
# DATASET LOADER FOR OPEN IMAGES DATASET (OID) WITH XML ANNOTATIONS
# ============================================================================

class OIDDetectionDataset(Dataset):
    """
    Dataset loader for Open Images Dataset v7 with Pascal VOC XML annotations.
    
    Expected XML format:
    <annotation>
        <filename>person_001.jpg</filename>
        <size>
            <width>640</width>
            <height>480</height>
        </size>
        <object>
            <name>Person</name>
            <bndbox>
                <xmin>100</xmin>
                <ymin>150</ymin>
                <xmax>300</xmax>
                <ymax>400</ymax>
            </bndbox>
        </object>
    </annotation>
    """
    
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir: Path to datasets/part_a/
            mode: 'train' or 'validation'
            transform: Augmentation transforms
        """
        self.root_dir = Path(root_dir) / mode
        self.transform = transform
        
        # Class mapping: Background (0), Human (1), Animal (2)
        # Support both capitalized and lowercase class names
        self.class_mapping = {
            'Person': 1, 'person': 1,  # Human class
            'Dog': 2, 'dog': 2,        # Animal classes
            'Cat': 2, 'cat': 2,
            'Horse': 2, 'horse': 2,
            'Elephant': 2, 'elephant': 2,
            'Bear': 2, 'bear': 2,
            'Zebra': 2, 'zebra': 2,
            'Giraffe': 2, 'giraffe': 2
        }
        
        self.samples = []
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples from {mode} set")
    
    def _load_samples(self):
        """Scan directory and collect all image-XML annotation pairs."""
        print(f"Scanning {self.root_dir}...")
        
        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Normalize class name (capitalize first letter)
            class_name_normalized = class_name.capitalize()
            
            # Check if class is in our mapping (case-insensitive)
            matched_class = None
            for key in self.class_mapping.keys():
                if key.lower() == class_name.lower():
                    matched_class = key
                    break
            
            if matched_class is None:
                print(f"⚠️  Skipping {class_name}: not in class mapping")
                continue
            
            # Find images and XML annotations
            img_dir = class_dir / 'images'
            
            # Try different possible directory names for annotations
            xml_dir = None
            for possible_name in ['pascal', 'Pascal', 'annotations', 'Annotations']:
                potential_dir = class_dir / possible_name
                if potential_dir.exists():
                    xml_dir = potential_dir
                    break
            
            if not img_dir.exists():
                print(f"⚠️  Skipping {class_name}: missing images/ directory")
                continue
            
            if xml_dir is None:
                print(f"⚠️  Skipping {class_name}: missing pascal/ or annotations/ directory")
                continue
            
            # Collect all image-XML pairs
            img_files = list(img_dir.glob('*.jpg'))
            print(f"  Found {len(img_files)} images in {class_name}")
            
            found_pairs = 0
            for img_path in img_files:
                xml_path = xml_dir / f"{img_path.stem}.xml"
                if xml_path.exists():
                    self.samples.append({
                        'image_path': img_path,
                        'xml_path': xml_path,
                        'class_name': matched_class
                    })
                    found_pairs += 1
            
            print(f"  Loaded {found_pairs} image-XML pairs from {class_name}")
            
            if found_pairs == 0:
                print(f"  ⚠️  No matching XML files found for images")
                # Show first few image names and check for XML
                for img_path in list(img_files)[:3]:
                    xml_path = xml_dir / f"{img_path.stem}.xml"
                    print(f"    Looking for: {xml_path.name} - Exists: {xml_path.exists()}")
    
    def _parse_xml(self, xml_path):
        """
        Parse Pascal VOC XML annotation file.
        
        Returns:
            boxes: List of [xmin, ymin, xmax, ymax]
            labels: List of class IDs
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            boxes = []
            labels = []
            
            # Parse each object in the annotation
            for obj in root.findall('object'):
                # Try different tag names for class name
                name_elem = obj.find('name')
                if name_elem is None:
                    name_elem = obj.find('n')  # Some XMLs use 'n' instead of 'name'
                
                if name_elem is None:
                    continue
                
                name = name_elem.text
                if name is None:
                    continue
                
                # Normalize class name (capitalize first letter)
                name = name.capitalize()
                
                if name not in self.class_mapping:
                    # Try lowercase match
                    for key in self.class_mapping.keys():
                        if key.lower() == name.lower():
                            name = key
                            break
                
                if name not in self.class_mapping:
                    continue
                
                # Get bounding box
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue
                
                xmin_elem = bndbox.find('xmin')
                ymin_elem = bndbox.find('ymin')
                xmax_elem = bndbox.find('xmax')
                ymax_elem = bndbox.find('ymax')
                
                if any(e is None for e in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                    continue
                
                try:
                    xmin = float(xmin_elem.text)
                    ymin = float(ymin_elem.text)
                    xmax = float(xmax_elem.text)
                    ymax = float(ymax_elem.text)
                except (ValueError, TypeError):
                    continue
                
                # Validate box
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(self.class_mapping[name])
            
            return boxes, labels
        
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return [], []
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['image_path']))
        if image is None:
            raise ValueError(f"Cannot load image: {sample['image_path']}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w = image.shape[:2]
        
        # Parse XML annotations
        boxes, labels = self._parse_xml(sample['xml_path'])
        
        # Handle empty annotations (fallback)
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]  # Background class
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        # Apply transforms
        if self.transform:
            try:
                transformed = self.transform(
                    image=image,
                    bboxes=boxes,
                    labels=labels
                )
                image = transformed['image']
                boxes = np.array(transformed['bboxes'], dtype=np.float32)
                labels = np.array(transformed['labels'], dtype=np.int64)
            except Exception as e:
                print(f"Transform error for {sample['image_path']}: {e}")
                # Fallback to no transform
                image = transforms.ToTensor()(image)
        else:
            image = transforms.ToTensor()(image)
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dict for Faster R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        return image, target


class OIDClassificationDataset(Dataset):
    """
    Dataset for classification (Human vs Animal).
    Extracts cropped objects from XML bounding box annotations.
    """
    
    def __init__(self, root_dir, mode='train', transform=None):
        """
        Args:
            root_dir: Path to datasets/part_a/
            mode: 'train' or 'validation'
            transform: Augmentation transforms
        """
        self.root_dir = Path(root_dir) / mode
        self.transform = transform
        self.samples = []
        
        # Class mapping: 0=Human, 1=Animal
        # Support both capitalized and lowercase
        self.class_mapping = {
            'Person': 0, 'person': 0,     # Human
            'Dog': 1, 'dog': 1,           # Animals
            'Cat': 1, 'cat': 1,
            'Horse': 1, 'horse': 1,
            'Elephant': 1, 'elephant': 1,
            'Bear': 1, 'bear': 1,
            'Zebra': 1, 'zebra': 1,
            'Giraffe': 1, 'giraffe': 1
        }
        
        self._load_samples()
        print(f"Loaded {len(self.samples)} classification samples from {mode} set")
    
    def _load_samples(self):
        """Extract all bounding box regions as individual samples."""
        for class_dir in self.root_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            
            # Check if class is in mapping (case-insensitive)
            matched_class = None
            for key in self.class_mapping.keys():
                if key.lower() == class_name.lower():
                    matched_class = key
                    break
            
            if matched_class is None:
                continue
            
            img_dir = class_dir / 'images'
            
            # Try different annotation directory names
            xml_dir = None
            for possible_name in ['pascal', 'Pascal', 'annotations', 'Annotations']:
                potential_dir = class_dir / possible_name
                if potential_dir.exists():
                    xml_dir = potential_dir
                    break
            
            if not img_dir.exists() or xml_dir is None:
                continue
            
            # Process each image-XML pair
            for img_path in img_dir.glob('*.jpg'):
                xml_path = xml_dir / f"{img_path.stem}.xml"
                if not xml_path.exists():
                    continue
                
                # Parse XML to get bounding boxes
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    for obj in root.findall('object'):
                        # Try different tag names
                        name_elem = obj.find('name')
                        if name_elem is None:
                            name_elem = obj.find('n')
                        
                        if name_elem is None or name_elem.text is None:
                            continue
                        
                        name = name_elem.text
                        
                        # Match class (case-insensitive)
                        matched = None
                        for key in self.class_mapping.keys():
                            if key.lower() == name.lower():
                                matched = key
                                break
                        
                        if matched is None:
                            continue
                        
                        # Get bounding box
                        bndbox = obj.find('bndbox')
                        if bndbox is None:
                            continue
                        
                        try:
                            xmin = int(float(bndbox.find('xmin').text))
                            ymin = int(float(bndbox.find('ymin').text))
                            xmax = int(float(bndbox.find('xmax').text))
                            ymax = int(float(bndbox.find('ymax').text))
                        except (ValueError, TypeError, AttributeError):
                            continue
                        
                        # Validate box
                        if xmax > xmin and ymax > ymin:
                            self.samples.append({
                                'image_path': img_path,
                                'bbox': [xmin, ymin, xmax, ymax],
                                'label': self.class_mapping[matched]
                            })
                
                except Exception as e:
                    continue
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['image_path']))
        if image is None:
            # Return dummy data if image can't be loaded
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, sample['label']
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Crop bounding box region
        x1, y1, x2, y2 = sample['bbox']
        
        # Ensure valid crop coordinates
        img_h, img_w = image.shape[:2]
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))
        
        cropped = image[y1:y2, x1:x2]
        
        # Handle edge cases
        if cropped.size == 0 or cropped.shape[0] == 0 or cropped.shape[1] == 0:
            cropped = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Apply transforms
        if self.transform:
            try:
                transformed = self.transform(image=cropped)
                cropped = transformed['image']
            except Exception as e:
                print(f"Transform error: {e}")
                # Fallback transform
                cropped = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])(cropped)
        else:
            cropped = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(cropped)
        
        label = sample['label']
        
        return cropped, label


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def get_detection_transforms(mode='train'):
    """Augmentation for detection task."""
    if mode == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
            A.Rotate(limit=10, p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))
    else:
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_classification_transforms(mode='train'):
    """Augmentation for classification task."""
    if mode == 'train':
        return A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.GaussNoise(var_limit=(10, 30), p=0.2),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def collate_fn(batch):
    """Custom collate function for detection."""
    return tuple(zip(*batch))


def train_detection_model(train_dataset, val_dataset, epochs=5, batch_size=1, lr=0.001):
    """
    Train Faster R-CNN detection model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"TRAINING DETECTION MODEL (Faster R-CNN)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}\n")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Build model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, 
        num_classes=3  # Background (0) + Human (1) + Animal (2)
    )
    model.to(device)
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    
    # Initialize wandb
    try:
        wandb.init(project="oid-detection-training", name=f"faster-rcnn-{epochs}epochs")
        use_wandb = True
    except:
        print("Warning: wandb not available, continuing without logging")
        use_wandb = False
    
    best_loss = float('inf')
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, targets in progress_bar:
            try:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += losses.item()
                num_batches += 1
                progress_bar.set_postfix({'loss': f"{losses.item():.4f}"})
            
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        if num_batches == 0:
            print("No valid batches in this epoch!")
            continue
        
        avg_loss = epoch_loss / num_batches
        
        # Validation
        model.train()  # Keep in train mode for loss calculation
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                try:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
                    val_batches += 1
                except Exception as e:
                    continue
        
        val_loss = val_loss / max(val_batches, 1)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'models/detection_model.pth')
            print(f"✅ Best model saved! Val Loss: {val_loss:.4f}\n")
        
        lr_scheduler.step()
    
    if use_wandb:
        wandb.finish()
    
    print(f"\n{'='*70}")
    print("DETECTION MODEL TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best model saved to: models/detection_model.pth")
    print(f"Best validation loss: {best_loss:.4f}\n")


def train_classification_model(train_dataset, val_dataset, epochs=5, batch_size=1, lr=0.001):
    """
    Train ResNet50 classification model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"TRAINING CLASSIFICATION MODEL (ResNet50)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}\n")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Build model
    model = resnet50(pretrained=True)
    
    # Freeze early layers
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Human (0), Animal (1)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Initialize wandb
    try:
        wandb.init(project="oid-classification-training", name=f"resnet50-{epochs}epochs")
        use_wandb = True
    except:
        print("Warning: wandb not available, continuing without logging")
        use_wandb = False
    
    best_acc = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100*correct/total:.2f}%"
            })
        
        train_acc = 100 * correct / total
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/classification_model.pth')
            print(f"✅ Best model saved! Val Acc: {val_acc:.2f}%\n")
        
        lr_scheduler.step()
    
    if use_wandb:
        wandb.finish()
    
    print(f"\n{'='*70}")
    print("CLASSIFICATION MODEL TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"Best model saved to: models/classification_model.pth")
    print(f"Best validation accuracy: {best_acc:.2f}%\n")


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================
def model_exists(path):
    return Path(path).exists()

def model_exists(path):
    return Path(path).exists()


def main():
    print("="*70)
    print("TRAINING MODELS ON OPEN IMAGES DATASET (OID v7) - XML VERSION")
    print("="*70)

    # ------------------------------------------------------------------
    # Create models directory
    # ------------------------------------------------------------------
    Path('models').mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Check dataset existence
    # ------------------------------------------------------------------
    data_path = Path('datasets/part_a')
    if not data_path.exists():
        print("❌ Error: datasets/part_a/ not found!")
        return

    train_path = data_path / 'train'
    if not train_path.exists():
        print("❌ Error: datasets/part_a/train/ not found!")
        return

    # ------------------------------------------------------------------
    # PART 1: DETECTION MODEL
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("PART 1: DETECTION MODEL")
    print("="*70)

    if model_exists("models/detection_model.pth"):
        print("✅ Detection model already exists. Skipping detection training.")
    else:
        print("\n📦 Loading detection datasets...")

        train_detection = OIDDetectionDataset(
            root_dir='datasets/part_a',
            mode='train',
            transform=get_detection_transforms('train')
        )

        val_detection = OIDDetectionDataset(
            root_dir='datasets/part_a',
            mode='validation',
            transform=get_detection_transforms('val')
        )

        if len(train_detection) == 0:
            print("❌ No detection training samples found!")
            return

        print("✅ Detection datasets loaded!")

        train_detection_model(
            train_dataset=train_detection,
            val_dataset=val_detection,
            epochs=5,
            batch_size=1,
            lr=0.001
        )

    # ------------------------------------------------------------------
    # PART 2: CLASSIFICATION MODEL
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("PART 2: CLASSIFICATION MODEL")
    print("="*70)

    if model_exists("models/classification_model.pth"):
        print("✅ Classification model already exists. Skipping classification training.")
    else:
        print("\n📦 Loading classification datasets...")

        train_classification = OIDClassificationDataset(
            root_dir='datasets/part_a',
            mode='train',
            transform=get_classification_transforms('train')
        )

        val_classification = OIDClassificationDataset(
            root_dir='datasets/part_a',
            mode='validation',
            transform=get_classification_transforms('val')
        )

        if len(train_classification) == 0:
            print("❌ No classification training samples found!")
            return

        print("✅ Classification datasets loaded!")

        train_classification_model(
            train_dataset=train_classification,
            val_dataset=val_classification,
            epochs=5,
            batch_size=1,
            lr=0.001
        )

    # ------------------------------------------------------------------
    # DONE
    # ------------------------------------------------------------------
    print("\n" + "="*70)
    print("🎉 TRAINING PIPELINE COMPLETE")
    print("="*70)
    print("Saved models:")
    print("  ✅ models/detection_model.pth")
    print("  ✅ models/classification_model.pth")
    print("\nYou can now run inference using:")
    print("  python main.py")
    print("="*70)



if __name__ == "__main__":
    main()