import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
from torch import nn

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 ResNet18 模型
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("cat_dog_model_resnet18.pth", map_location=device))
model = model.to(device)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 猫狗标签
classes = ["cat", "dog"]

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    # 推理 & 输出
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        predicted = torch.argmax(probs, dim=1).item()

    print(f"预测结果：这张图是 -> {classes[predicted]}，置信度：{probs[0][predicted]:.2f}")
    print(f"完整概率分布: {classes[0]}={probs[0][0]:.2f}, {classes[1]}={probs[0][1]:.2f}")

# 示例调用
predict_image("data/cats_and_dogs_filtered/validation/dogs/dog.2000.jpg")
