import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torchvision.models import resnet18

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用 ResNet18 模型
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
model.load_state_dict(torch.load("cat_dog_model_resnet18.pth", map_location=device))
model.eval()

# 标签
classes = ["cat", "dog"]

# 图像预处理方式
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# 定义预测函数
def classify_image(image):
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted].item()

    return {classes[0]: float(probs[0][0]), classes[1]: float(probs[0][1])}

# 启动界面
gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="🐾 猫狗图像分类器（ResNet18版）",
    description="上传一张图片，我来猜猜是猫还是狗！",
).launch(share=True) # 允许生成一个公开的链接