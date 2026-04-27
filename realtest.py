import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# same transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth"))

model.eval()

# load image
img = Image.open("images/person.jpg").convert("RGB")
img = transform(img).unsqueeze(0)

# predict

with torch.no_grad():
    output = model(img)
    pred = torch.argmax(output, 1)

print("it is AI" if pred.item() == 0 else "It is Real")