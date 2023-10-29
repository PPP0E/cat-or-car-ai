import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import sys

def process_image(image_path):
    transform_pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path)
    return transform_pipeline(image).unsqueeze(0)


def guess(image_tensor, model):
    device = "mps:0"
    model.to(device)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)
    return predicted.item()


def classify_image(class_idx):
    if class_idx in [281, 282, 283]: return "cat";
    elif class_idx == 817: return "car";
    return "something else"

image_tensor = process_image(str(sys.argv[1]).strip())
class_idx = guess(image_tensor, resnet50(weights=ResNet50_Weights.DEFAULT).eval())
result = classify_image(class_idx)
print(f"It's a {result}. ({class_idx})")
