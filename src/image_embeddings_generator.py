from torchvision import models, transforms
from PIL import Image
import torch

# Pretrained model
model = models.resnet50(pretrained=True)
model.eval()

# Remove final classification layer
model = torch.nn.Sequential(*list(model.children())[:-1])

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_image_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image_tensor)
    return features.squeeze().numpy()




