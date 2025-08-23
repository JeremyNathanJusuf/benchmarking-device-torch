import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time
import timm
import torch.nn as nn

# Detect device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Benchmark parameters
batch_size = 64
epochs = 1
num_workers = 2 if device.type == 'cuda' else 0

# Models to benchmark
model_names = ['resnet18', 'resnet50', 'vit_base_patch16_224']

# Dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_set = datasets.FakeData(transform=transform)
test_set = datasets.FakeData(transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Benchmark loop
for model_name in model_names:
    print(f'\nBenchmarking model: {model_name}')
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Training
        start_train = time.time()
        model.train()
        for _ in range(epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        end_train = time.time()

        # Inference
        model.eval()
        total = 0
        start_inf = time.time()
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                total += inputs.size(0)
        end_inf = time.time()

        # Results
        train_time = end_train - start_train
        inf_speed = total / (end_inf - start_inf)
        vram_peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == 'cuda' else 0

        print(f'Device: {device.type.upper()}')
        print(f'Training time: {train_time:.2f}s')
        print(f'Inference speed: {inf_speed:.2f} images/sec')
        print(f'Peak VRAM usage: {vram_peak:.2f} GB')

    except RuntimeError as e:
        print(f'Model {model_name} failed: {str(e).splitlines()[0]}')

