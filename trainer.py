import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# resize the dataset images and convert them into tensors (tensors are 4d array da)

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# a line to transform data images

dataset = datasets.ImageFolder(
    "dataset",
    transform=transform
)

# Feed the models batchwise

loader = DataLoader(dataset, batch_size=32,shuffle=True)

# 2. CREATE MODEL (missing in your code)

model = models.resnet18(pretrained=True)


model.fc = nn.Linear(model.fc.in_features, 2)

# 3. loss

loss_fn = nn.CrossEntropyLoss()

# 4. optimizer

optimizer = optim.Adam(model.parameters(), lr=0.001)

for round in range(5):
    for images, labels in loader:

        # 1. not ai comment da: Prediction
        outputs = model(images)

        # 2. not ai comment da: Calculate error
        loss = loss_fn(outputs, labels)

        # 3. not ai comment da: Clear old gradients
        optimizer.zero_grad()

        # 4. not ai comment da: Backpropagation
        loss.backward()

        # 5. not ai comment da: Update weights
        optimizer.step()

    print(f"Round {round}, Loss: {loss.item()}")

torch.save(model.state_dict(), "model.pth")

print("Model saved successfully")