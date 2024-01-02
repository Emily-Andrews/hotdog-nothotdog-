
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm for progress bars

# Define transforms and data loader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Assuming your data is in a folder structure with 'train' and 'val' subfolders
train_dataset = torchvision.datasets.ImageFolder('~/code/hotdog-nothotdog/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

val_dataset = torchvision.datasets.ImageFolder('~/code/hotdog-nothotdog/test', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define the ResNet-50 model for binary classification
class BinaryResNet50(nn.Module):
    def __init__(self):
        super(BinaryResNet50, self).__init__()
        resnet = resnet50(weights=True)
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Add a new fully connected layer for binary classification
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)

# Instantiate the model, loss function, and optimizer
model = BinaryResNet50()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    # Use tqdm for the training loop progress bar
    with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as train_bar:
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += labels.size(0)
            train_correct += ((outputs > 0.5).float() == labels.float().view(-1, 1)).sum().item()

            # Update the progress bar description with current loss and accuracy
            train_bar.set_postfix({'Loss': train_loss / train_total, 'Accuracy': train_correct / train_total})

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.float().view(-1, 1))

            val_loss += loss.item()
            total += labels.size(0)
            correct += ((outputs > 0.5).float() == labels.float().view(-1, 1)).sum().item()

    accuracy = correct / total
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}, Validation Accuracy: {accuracy:.2%}')

#Save the trained model
torch.save(model.state_dict(), 'binary_resnet50.pth')
