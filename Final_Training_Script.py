import os
import sys
import sklearn
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, r2_score, mean_squared_error
from tqdm import tqdm
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Dataset Validation Utility Functions
class DatasetError(Exception):
    """Custom exception for dataset-related errors"""
    pass


def validate_dataset_path(dataset_path):
    """Validate and print details about the dataset path"""
    logging.info(f"Checking dataset path: {dataset_path}")
    abs_dataset_path = os.path.abspath(dataset_path)
    if not os.path.exists(abs_dataset_path):
        raise FileNotFoundError(f"Dataset directory not found: {abs_dataset_path}")
    if not os.path.isdir(abs_dataset_path):
        raise NotADirectoryError(f"Path is not a directory: {abs_dataset_path}")
    contents = os.listdir(abs_dataset_path)
    logging.info("Dataset directory contents: %s", contents)
    subdirs = [d for d in contents if os.path.isdir(os.path.join(abs_dataset_path, d))]
    logging.info("Subdirectories: %s", subdirs)
    return abs_dataset_path


def load_dataset_safely(dataset_path):
    """Safely load the dataset with detailed error reporting"""
    try:
        validated_path = validate_dataset_path(dataset_path)
        dataset = ImageFolder(root=validated_path)
        return dataset
    except Exception as e:
        logging.error(f"Dataset loading failed: {e}")
        raise DatasetError(f"Could not load dataset from {dataset_path}") from e


# Freshness Dataset Class (using ImageFolder directly)
class FreshnessDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root=root_dir, transform=transform)
        self.class_names = self.dataset.classes
        logging.info(f"Loaded {len(self.dataset)} images from {len(self.class_names)} classes")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        item_name = self.class_names[label]  # Get item name from class name
        return image, label, item_name
    
# Freshness Model Definition
class FreshnessModel(nn.Module):
    def __init__(self, num_classes, backbone="resnet18"):
        super(FreshnessModel, self).__init__()

        if backbone == "resnet18":
            self.base_model = models.resnet18(pretrained=True)
            self.feature_dim = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(self.feature_dim, num_classes)

        elif backbone == "efficientnet_b0":
            self.base_model = models.efficientnet_b0(pretrained=True)
            self.feature_dim = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(self.feature_dim, num_classes),
            )
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.base_model(x)
    
    def extract_features(self, x):
        """Extract features before classification layer"""
        if isinstance(self.base_model, models.ResNet):
            # For ResNet
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)

            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)

            x = self.base_model.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        else:
            # For EfficientNet
            x = self.base_model.features(x)
            x = self.base_model._avg_pooling(x)
            x = x.flatten(start_dim=1)
            x = self.base_model._dropout(x)
            return x

# Freshness Model Training Function
def train_freshness_model(dataset_path, batch_size=32, num_epochs=10, backbone="resnet18", save_path="freshness_model.pth", augment_data=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if augment_data:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transform

    # Determine number of classes dynamically
    base_dataset = FreshnessDataset(root_dir=dataset_path, transform=transform)
    num_classes = len(base_dataset.class_names)
    
    print(f"Freshness classes detected: {base_dataset.class_names}")
    print(f"Number of freshness classes: {num_classes}")

    # Initialize the model with the correct number of classes
    model = FreshnessModel(num_classes, backbone=backbone)

    # Split the dataset
    train_size = int(0.8 * len(base_dataset))
    test_size = len(base_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(base_dataset, [train_size, test_size])

    # Apply augmentation ONLY to the training data
    train_dataset.dataset.transform = train_transform if augment_data else transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize the model, loss function, and optimizer
    model = FreshnessModel(num_classes, backbone=backbone)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    print(f"Training freshness model on device: {device}")

    # Training loop
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Train Accuracy: {accuracy:.2f}%")
        
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            # When saving, also save the class names
            torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': base_dataset.class_names
        }, save_path)

        return model, base_dataset.class_names

# Shelf Life Model Definitions
class ShelfLifeModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(ShelfLifeModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 32)
        self.fc3 = nn.Linear(32, 1)  # Output is shelf life (a single value)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
    
def extract_shelf_life_features(produce_type, freshness, storage_conditions):
    """
    Extract comprehensive features for shelf life prediction
    """
    # Base shelf life for different produce types (in days)
    base_shelf_life = {
        'apple': 30,
        'banana': 7,
        'strawberry': 3,
        'tomato': 5,
        'cucumber': 7,
        'carrot': 21,
        'bellpepper': 7,
        'mango': 5,
        'orange': 14,
        'potato': 60
    }
    
    # Factors affecting shelf life
    features = [
        base_shelf_life.get(produce_type.lower(), 10),  # Base shelf life
        freshness,  # Freshness score
        storage_conditions.get('temperature', 20),  # Storage temperature
        storage_conditions.get('humidity', 50),  # Humidity
        storage_conditions.get('light_exposure', 0),  # Light exposure
        1 if storage_conditions.get('refrigerated', False) else 0,  # Refrigeration
        storage_conditions.get('packaging_type', 0),  # Packaging quality
    ]
    
    return np.array(features)
        
def create_shelf_life_dataset(num_samples=1000):
    """
    Generate a more realistic synthetic dataset for shelf life prediction
    """
    # List of produce types
    produce_types = ['apple', 'banana', 'strawberry', 'tomato', 'cucumber', 
                     'carrot', 'bellpepper', 'mango', 'orange', 'potato']
    
    # Base shelf life dictionary
    base_shelf_life = {
        'apple': 30,
        'banana': 10,
        'strawberry': 3,
        'tomato': 5,
        'cucumber': 7,
        'carrot': 21,
        'bellpepper': 7,
        'mango': 5,
        'orange': 14,
        'potato': 60
    }
    
    X = []
    y = []
    
    for _ in range(num_samples):
        # Randomly select produce type
        produce_type = np.random.choice(produce_types)
        
        # Generate storage conditions
        storage_conditions = {
            'temperature': np.random.uniform(0, 30),  # 0-30°C
            'humidity': np.random.uniform(30, 80),  # 30-80%
            'light_exposure': np.random.uniform(0, 1),  # 0-1 scale
            'refrigerated': np.random.choice([True, False]),
            'packaging_type': np.random.randint(0, 3)  # Different packaging qualities
        }
        
        # Generate freshness score
        freshness = np.random.uniform(0, 1)
        
        # Extract features
        features = extract_shelf_life_features(produce_type, freshness, storage_conditions)
        
        # Calculate shelf life with some randomness
        # Use the base_shelf_life dictionary directly
        base_life = base_shelf_life.get(produce_type.lower(), 10)
        temp_factor = 1 - (features[2] / 30)  # Higher temp reduces shelf life
        humidity_factor = 1 - (features[3] / 100)  # Higher humidity reduces shelf life
        freshness_factor = features[1]
        refrigeration_bonus = 1.5 if features[5] == 1 else 1
        shelf_life = (base_life * freshness_factor * temp_factor * 
                      humidity_factor * refrigeration_bonus + 
                      np.random.normal(0, 2))  # Add some noise
        
        X.append(features)
        y.append(max(0, shelf_life))  # Ensure non-negative shelf life
    
    return np.array(X), np.array(y)

def train_shelf_life_model(X=None, y=None, model_save_path="shelf_life_model.pth"):
    """
    Train a more sophisticated shelf life prediction model
    """
    print("\nTraining advanced shelf life prediction model...")
    
    # If no data provided, generate synthetic dataset
    if X is None or y is None:
        X, y = create_shelf_life_dataset()
    
    # Ensure NumPy arrays
    X = np.array(X)
    y = np.array(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Try multiple models
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    best_model = None
    best_score = float('-inf')
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"{name} Model:")
        print(f"Train R^2 score: {train_score:.4f}")
        print(f"Test R^2 score: {test_score:.4f}")
        
        # Update best model
        if test_score > best_score:
            best_model = model
            best_score = test_score
    
    # Save the model and scaler
    torch.save({
        'model': best_model,
        'scaler': scaler,
        'feature_names': [
            'base_shelf_life', 'freshness', 'temperature', 
            'humidity', 'light_exposure', 'refrigerated', 
            'packaging_type'
        ]
    }, model_save_path)
    
    print(f"Shelf life model saved to {model_save_path}")
    
    return best_model, scaler

# Combined Model for Feature Extraction
class CombinedModel(nn.Module):
    def __init__(self, model_list):
        super(CombinedModel, self).__init__()
        self.models = nn.ModuleList(model_list)
        
    def forward(self, x):
        features = []
        for model in self.models:
            features.append(model(x))
        return torch.cat(features, dim=1)

class CustomImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        # Custom initialization to handle nested directory structure
        self.root_dir = root_dir
        self.transform = transform
        self.produce_classes = []
        self.freshness_classes = ['fresh', 'rotten']
        self.samples = []
        self.custom_class_to_idx = {}
        
        # Collect all unique produce types and build samples
        for produce_type in os.listdir(root_dir):
            produce_path = os.path.join(root_dir, produce_type)
            if os.path.isdir(produce_path):
                self.produce_classes.append(produce_type)
                
                # Collect images for each freshness state
                for freshness in self.freshness_classes:
                    freshness_path = os.path.join(produce_path, freshness)
                    if os.path.isdir(freshness_path):
                        # Create unique class name
                        class_name = f"{produce_type}_{freshness}"
                        class_idx = len(self.custom_class_to_idx)
                        self.custom_class_to_idx[class_name] = class_idx
                        
                        # Collect image paths
                        for img_name in os.listdir(freshness_path):
                            img_path = os.path.join(freshness_path, img_name)
                            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                                self.samples.append((img_path, class_idx))
        
        # Set classes and class_to_idx
        self.classes = list(self.custom_class_to_idx.keys())
        self.class_to_idx = self.custom_class_to_idx
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
        print("Produce Classes:", self.produce_classes)
        print("Freshness Classes:", self.freshness_classes)
        print("Full Classes:", self.classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Open and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Fruits and Vegetables Classifier Trainer
class FruitsVegetablesTrainer:
    def __init__(self, dataset_path='fruits_vegetables_dataset', 
                 batch_size=32, model_save_path='best_fruits_model.pth'):
        """
        Initialize the trainer with dataset path and configuration
        """
        # Validate and get absolute path
        self.dataset_path = validate_dataset_path(dataset_path)
        self.batch_size = batch_size
        self.model_save_path = model_save_path
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nTraining produce classifier using device: {self.device}")
        
        # Initialize models and data
        self.load_models()
        self.create_data_loaders()
    
    def load_models(self):
        """
        Load and prepare base models
        """
    # EfficientNet-B3
        efficientnet = models.efficientnet_b3(weights='IMAGENET1K_V1')
        efficientnet.classifier = nn.Identity()  # Remove classification layer
    
    # ResNet-50
        resnet50 = models.resnet50(weights='IMAGENET1K_V1')
        resnet50.fc = nn.Identity()
    
        # Combined models dictionary
        self.models_dict = {
            'efficientnet': efficientnet,
            'resnet50': resnet50,
            'combined': CombinedModel([efficientnet, resnet50])
        }
    
    # Move models to device
        for model in self.models_dict.values():
            model.to(self.device)
    
    def create_data_loaders(self):
        """
    Prepare data loaders for training and validation
    """
    # Data preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
        ])

    # Load dataset
        try:
            # Use CustomImageFolder with explicit root argument
            dataset = CustomImageFolder(root_dir=self.dataset_path, transform=transform)
    
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
        # Define custom collate function
            def custom_collate(batch):
                images = [item[0] for item in batch]
                labels = [item[1] for item in batch]
                return torch.stack(images), torch.tensor(labels)
    
            self.train_loader = DataLoader(train_dataset, 
                                   batch_size=self.batch_size, 
                                   shuffle=True, 
                                   collate_fn=custom_collate)
            self.val_loader = DataLoader(val_dataset, 
                                 batch_size=self.batch_size, 
                                 shuffle=False, 
                                 collate_fn=custom_collate)
    
        # Set produce classes and full classes
            self.produce_classes = dataset.produce_classes  # This will now be ['Apple', 'Banana', etc.]
            self.freshness_classes = dataset.freshness_classes
        
        # Create class names that include both produce and freshness
            self.class_names = [f"{produce}_{freshness}" 
                            for produce in self.produce_classes 
                            for freshness in self.freshness_classes]
        
            print(f"Loaded {len(dataset)} images from {len(self.produce_classes)} produce types")
            print(f"Produce classes: {self.produce_classes}")
            print(f"Freshness classes: {self.freshness_classes}")
            print(f"Full class names: {self.class_names}")
    
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to mock dataset for demonstration")
            # Create mock data for demonstration
            self.produce_classes = ['Apple', 'Banana','Bellpepper','Carrot','Cucumber','Mango','Orange', 'Potato','Strawberry','Tomato']
            self.freshness_classes = ['fresh', 'rotten']
            self.class_names = [f"{produce}_{freshness}" 
                            for produce in self.produce_classes 
                            for freshness in self.freshness_classes]
            self.train_loader = None
            self.val_loader = None
        
    def extract_features(self, model, data_loader):
        """
        Extract features from a given model and data loader
    """
        if data_loader is None:
            # Generate mock features for demonstration
            print("Generating mock features for demonstration")
            features = np.random.randn(100, 2048)  # Mock features
            labels = np.random.randint(0, len(self.class_names), 100)  # Mock labels
            return features, labels
    
        features = []
        labels = []
        model.eval()

        with torch.no_grad():
            for inputs, targets in tqdm(data_loader, desc="Extracting features"):
                inputs = inputs.to(self.device)
            
                # Handle different model types
                if isinstance(model, CombinedModel):
                    # For combined model, extract features from each sub-model
                    outputs = model(inputs)
                else:
                    # For single models
                    outputs = model(inputs)
            
                features.append(outputs.cpu().numpy())
                labels.append(targets.cpu().numpy())
    
        return np.vstack(features), np.concatenate(labels)
    
    def train_and_evaluate(self):
        """
        Train and evaluate different model configurations
        """
        results = {}
        best_model_name = None
        best_model = None
        best_accuracy = 0
        best_classifier = None
        best_scaler = None
        
        for model_name, model in self.models_dict.items():
            print(f"Evaluating {model_name}...")
            
            # Extract features
            X_train, y_train = self.extract_features(model, self.train_loader)
            X_val, y_val = self.extract_features(model, self.val_loader)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Initialize and train classifier
            clf = SVC(kernel='rbf', probability=True, random_state=42)
            clf.fit(X_train_scaled, y_train)
            
            # Predict and evaluate
            y_pred = clf.predict(X_val_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='weighted')
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            results[model_name] = metrics
            
            print(f"Results for {model_name}:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
            
            # Track the best model
            if metrics['accuracy'] > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_model_name = model_name
                best_model = model
                best_classifier = clf
                best_scaler = scaler
            
            print()
        
        # Save the best model and classifier
        torch.save({
        'model': best_model,
        'classifier': best_classifier,
        'scaler': best_scaler,
        'produce_classes': self.produce_classes,  # This will now be ['Apple', 'Banana', etc.]
        'freshness_classes': self.freshness_classes,
        'class_names': self.class_names,
        'best_model_name': best_model_name
        },  self.model_save_path)
        
        print(f"Best Performing Model: {best_model_name}")
        print(f"Accuracy: {best_accuracy:.4f}")
        print(f"Produce classifier saved to {self.model_save_path}\n")
        
        return best_model, best_classifier, best_scaler, best_model_name


# Integrated Training System
class FreshProduce:
    def __init__(self, args):
        """
        Initialize the integrated fresh produce analysis system
        """
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Paths for model saving
        self.freshness_model_path = os.path.join(args.output_dir, "freshness_model.pth")
        self.shelf_life_model_path = os.path.join(args.output_dir, "shelf_life_model.pth")
        self.produce_model_path = os.path.join(args.output_dir, "produce_model.pth")
        self.combined_model_path = os.path.join(args.output_dir, "combined_model.joblib")
        
        print("=" * 50)
        print("FRESH PRODUCE ANALYSIS SYSTEM")
        print("=" * 50)
        print(f"Output directory: {args.output_dir}")
        print(f"Device: {self.device}")
        
    def train_all_models(self):
        """
        Train all components of the system
        """
        # 1. Train freshness detection model
        print("\n" + "=" * 30)
        print("TRAINING FRESHNESS MODEL")
        print("=" * 30)
        freshness_model, freshness_classes = train_freshness_model(
            dataset_path=self.args.freshness_dataset,
            batch_size=self.args.batch_size,
            num_epochs=self.args.epochs,
            backbone=self.args.backbone,
            save_path=self.freshness_model_path
        )
        
        # 2. Train shelf life prediction model
        print("\n" + "=" * 30)
        print("TRAINING SHELF LIFE MODEL")
        print("=" * 30)

        # Generate/load shelf life training data
        if self.args.shelf_life_data:
        # Load actual shelf life data if available
            shelf_life_data = np.load(self.args.shelf_life_data)
            X_shelf = shelf_life_data['X']
            y_shelf = shelf_life_data['y']
        else:
            # Generate synthetic dataset
            X_shelf, y_shelf = create_shelf_life_dataset()

        shelf_life_model, shelf_life_scaler = train_shelf_life_model(
            X=X_shelf,
            y=y_shelf,
            model_save_path=self.shelf_life_model_path
        )
        
        # 3. Train produce classification model
        print("\n" + "=" * 30)
        print("TRAINING PRODUCE CLASSIFIER")
        print("=" * 30)
        fruits_trainer = FruitsVegetablesTrainer(
            dataset_path=self.args.produce_dataset,
            batch_size=self.args.batch_size,
            model_save_path=self.produce_model_path
        )
        
        produce_model, produce_classifier, produce_scaler, model_name = fruits_trainer.train_and_evaluate()
        
        # 4. Save the combined model
        self.save_combined_model(
            freshness_model=freshness_model,
            freshness_classes=freshness_classes,
            shelf_life_model=shelf_life_model,
            shelf_life_scaler=shelf_life_scaler,
            produce_model=produce_model,
            produce_classifier=produce_classifier,
            produce_scaler=produce_scaler,
            produce_classes=fruits_trainer.class_names
        )
    
    def save_combined_model(self, freshness_model, freshness_classes, 
                       shelf_life_model, shelf_life_scaler,
                       produce_model, produce_classifier, produce_scaler, produce_classes):
        """
        Save all trained models in a single joblib file
        """
        print("\n" + "=" * 30)
        print("SAVING COMBINED MODEL")
        print("=" * 30)
    
        # Ensure freshness classes are explicitly set
        if not freshness_classes:
            freshness_classes = ['fresh', 'rotten']
    
        # Store all components
        combined_model = {
            'freshness_model': freshness_model,
            'freshness_classes': freshness_classes,  # Explicitly save freshness classes
            'shelf_life_model': shelf_life_model,
            'shelf_life_scaler': shelf_life_scaler,
            'produce_model': produce_model,
            'produce_classifier': produce_classifier,
            'produce_scaler': produce_scaler,
            'produce_classes': produce_classes,
            'metadata': {
                'creation_date': np.datetime64('now'),
                'device': str(self.device),
                'backbone': self.args.backbone
            }
        }
    
        # Save to joblib
        joblib.dump(combined_model, self.combined_model_path)
        print(f"Successfully saved combined model to: {self.combined_model_path}")
        print("Freshness classes saved:", freshness_classes)
        print("\nTraining complete!")

def parse_args():
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='Fresh Produce Analysis System Training')
    
    # Dataset paths with default to script directory
    parser.add_argument('--freshness-dataset', type=str, 
                        default=os.path.join(script_dir, 'fruits_vegetables_dataset'),
                        help='Path to the freshness dataset')
    parser.add_argument('--produce-dataset', type=str, 
                        default=os.path.join(script_dir, 'fruits_vegetables_dataset'),
                        help='Path to the fruits and vegetables dataset')
    parser.add_argument('--shelf-life-data', type=str, default=None,
                        help='Path to shelf life data (.npz file with X and y arrays)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'efficientnet_b0'],
                        help='Backbone architecture for image models')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save trained models')
    
    return parser.parse_args()

def main():
    # Attempt to parse arguments with more error handling
    try:
        # Parse command-line arguments
        args = parse_args()
        
        # Additional path validation
        print("Validating dataset paths...")
        validate_dataset_path(args.freshness_dataset)
        validate_dataset_path(args.produce_dataset)
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create and train the system
        system = FreshProduce(args)
        system.train_all_models()
    
    except Exception as e:
        print("\n!!! CRITICAL ERROR !!!")
        print(f"An error occurred during setup: {e}")
        print("\nTroubleshooting tips:")
        print("1. Verify dataset paths")
        print("2. Ensure correct directory structure")
        print("3. Check file permissions")
        
        # Fallback to mock configuration if needed
        print("\nAttempting fallback configuration...")
        class Args:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            freshness_dataset = os.path.join(script_dir, 'fruits_vegetables_dataset')
            produce_dataset = os.path.join(script_dir, 'fruits_vegetables_dataset')
            shelf_life_data = None
            batch_size = 32
            epochs = 10
            backbone = 'resnet18'
            output_dir = 'models'
        
        args = Args()
        system = FreshProduce(args)
        system.train_all_models()

# Logging and system information
def log_system_info():
    """
    Log system and environment information
    """
    print("\nSYSTEM INFORMATION")
    print("=" * 20)
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
    print(f"NumPy Version: {np.__version__}")
    print(f"scikit-learn Version: {sklearn.__version__}")

# Entry point of the script
if __name__ == "__main__":
    # Log system information
    log_system_info()
    
    # Run the main training process
    main()