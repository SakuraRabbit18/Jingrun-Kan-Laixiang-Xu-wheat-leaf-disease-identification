import os
import time
import csv
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import os
import shutil
import random
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from LPNet import get_LPNet


from thop import profile
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
os.makedirs('results', exist_ok=True)

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes)}
        self.data = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                label = self.class_to_idx[class_name]
                self.data.append((img_path, label))

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)

class ModelTester:
    def __init__(self, model, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device

        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)

        self.results = {
            'model': model_name,
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': [],
            'test_acc': 0,
            'params': params,
            'flops': flops,
            'macs': flops / 2,
            'fps_gpu': 0,
            'fps_cpu': 0,
        }
    
    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Training {self.model_name}'):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_correct.double() / len(train_loader.dataset)
        
        self.results['train_loss'].append(epoch_loss)
        self.results['train_acc'].append(epoch_acc.item())
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        running_correct = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f'Validating {self.model_name}'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_correct.double() / len(val_loader.dataset)
        
        self.results['val_loss'].append(epoch_loss)
        self.results['val_acc'].append(epoch_acc.item())
        
        return epoch_loss, epoch_acc
    
    def test(self, test_loader, criterion):
        self.model.eval()
        running_correct = 0
        all_preds = []
        all_labels = []
        num_classes = len(test_loader.dataset.classes)
        class_names = test_loader.dataset.classes

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f'Testing {self.model_name}'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                running_correct += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = running_correct.double() / len(test_loader.dataset)
        self.results['test_acc'] = test_acc.item()

        cm = confusion_matrix(all_labels, all_preds)

        precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        sensitivity_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        specificity_list = []
        for i in range(cm.shape[0]):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            specificity_list.append(specificity)
        specificity_macro = np.mean(specificity_list)

        self.results['precision'] = precision_macro
        self.results['sensitivity'] = sensitivity_macro
        self.results['f1'] = f1_macro
        self.results['specificity'] = specificity_macro

        self.class_metrics = []
        for class_idx in range(num_classes):
            class_name = class_names[class_idx]
            class_total = cm[class_idx, :].sum()
            class_correct = cm[class_idx, class_idx]
            pred_total = cm[:, class_idx].sum()

            accuracy = class_correct / class_total if class_total > 0 else 0
            precision = class_correct / pred_total if pred_total > 0 else 0
            sensitivity = class_correct / class_total if class_total > 0 else 0
            tn = cm.sum() - (cm[class_idx, :].sum() + cm[:, class_idx].sum() - cm[class_idx, class_idx])
            fp = cm[:, class_idx].sum() - cm[class_idx, class_idx]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

            self.class_metrics.append({
                'class': class_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'F1-score': f1
            })

        return test_acc
    
    def measure_fps(self, input_size=(1, 3, 224, 224), warmup=10, repeat=100, device=None):
        if device is None:
            device = self.device
        model = self.model.to(device)
        dummy_input = torch.randn(input_size).to(device)

        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(repeat):
                if device == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                _ = model(dummy_input)
                if device == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start_time)

        avg_time = sum(times) / repeat
        fps = 1.0 / avg_time
        return fps

    def measure_and_record_fps(self):
        if torch.cuda.is_available():
            fps_gpu = self.measure_fps(device='cuda')
            self.results['fps_gpu'] = fps_gpu
        fps_cpu = self.measure_fps(device='cpu')
        self.results['fps_cpu'] = fps_cpu
    
    def save_results(self):
        df_train = pd.DataFrame({
            'epoch': range(1, len(self.results['train_acc'])+1),
            'train_acc': self.results['train_acc'],
            'val_acc': self.results['val_acc'],
            'train_loss': self.results['train_loss'],
            'val_loss': self.results['val_loss']
        })
        df_train.to_csv(f'results/{self.model_name}_training.csv', index=False)
        
        summary = {
            'model': self.model_name,
            'test_acc': self.results['test_acc'],
            'precision': self.results.get('precision', 0),
            'sensitivity': self.results.get('sensitivity', 0),
            'specificity': self.results.get('specificity', 0),
            'f1': self.results.get('f1', 0),
            'params': self.results['params'],
            'flops': self.results['flops'],
            'macs': self.results['macs'],
            'fps_gpu': self.results['fps_gpu'],
            'fps_cpu': self.results['fps_cpu'],
            'best_val_acc': max(self.results['val_acc']),
            'final_train_acc': self.results['train_acc'][-1]
        }
        file_path = 'results/summary_FullTest.csv'
        write_header = not os.path.exists(file_path)
        with open(file_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(summary)
        
        if self.class_metrics is not None:
            os.makedirs('results/class_metrics', exist_ok=True)
            class_df = pd.DataFrame(self.class_metrics)
            class_df = class_df.set_index('class')
            class_df.to_csv(f'results/class_metrics/{self.model_name}_class_metrics.csv')
            
        
        return summary

class ExperimentRunner:
    def __init__(self, num_classes, batch_size=64, num_epochs=25):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(int(224/0.875)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.train_dataset = CustomDataset(root_dir='Wheat_Disease_Dataset/train', transform=self.train_transform)
        self.val_dataset = CustomDataset(root_dir='Wheat_Disease_Dataset/val', transform=self.val_transform)
        self.test_dataset = CustomDataset(root_dir='Wheat_Disease_Dataset/test', transform=self.val_transform)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    def run_experiment(self, model, model_name):
        print(f"\n{'='*50}")
        print(f"Running experiment for {model_name}")
        print(f"{'='*50}")
        
        tester = ModelTester(model, model_name, self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        best_val_acc = 0.0
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            train_loss, train_acc = tester.train_epoch(self.train_loader, criterion, optimizer)
            val_loss, val_acc = tester.validate_epoch(self.val_loader, criterion)
            scheduler.step()
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"{model_name}_best.pth")
                print(f"Best model saved with accuracy: {best_val_acc:.4f}")
        
        test_acc = tester.test(self.test_loader, criterion)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        
        tester.measure_and_record_fps()
        
        summary = tester.save_results()
        print(f"\nResults saved for {model_name}")
        
        return summary


def LPNet(num_classes=5):
    model = get_LPNet()
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model



def main():
    num_classes = len(os.listdir('Wheat_Disease_Dataset/train'))
    runner = ExperimentRunner(num_classes=num_classes, num_epochs=200)
    
    models_to_test = [
        ('LpNet_ep200', LPNet),
    ]
    
    all_results = []
    for name, model_fn in models_to_test:
        model = model_fn(num_classes)
        results = runner.run_experiment(model, name)
        all_results.append(results)
    
    print("\n\n=== Final Results Summary ===")
    print(pd.DataFrame(all_results))

if __name__ == '__main__':
    main()