import os

def count_images(folder, exts=(".jpg", ".jpeg", ".png", ".webp")):
    count = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(exts):
                count += 1
    return count

# 替换为你的测试集路径
test_dir = 'data/cats_and_dogs_filtered/validation'


print("Total images in validation set:", count_images(test_dir))
