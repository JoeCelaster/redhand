# 1. imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

# resize the dataset images and convert them into tensors (tensors are 4d array )

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

# 2. CREATE MODEL (missing in your code ❌)

model = models.resnet18(pretrained=True)


model.fc = nn.Linear(model.fc.in_features, 2)

# 3. loss + optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for round in range(5):
    for images, labels in loader:

        # 1. Prediction
        outputs = model(images)

        # 2. Calculate error
        loss = loss_fn(outputs, labels)

        # 3. Clear old gradients
        optimizer.zero_grad()

        # 4. Backpropagation
        loss.backward()

        # 5. Update weights
        optimizer.step()

    print(f"Round {round}, Loss: {loss.item()}")
# ✅ ADD HERE
torch.save(model.state_dict(), "model.pth")

print("Model saved successfully")