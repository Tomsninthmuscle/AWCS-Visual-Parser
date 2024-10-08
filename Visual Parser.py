import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn

# 设置CUDA设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 定义sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 文件夹路径
folder_path = 'Your folder_path'


window_size = Numb_a
stride = numb_b


model.load_state_dict(torch.load('your_best_model.pt'))  
model.eval()  
model.to(device)  
# 定义图像变换
test_transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# 循环处理文件夹中的图片
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)

        test_image = Image.open(image_path)
        width, height = test_image.size


        heatmap = np.zeros((height, width), dtype=np.float32)


        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                # 提取子图
                sub_image = test_image.crop((x, y, x + window_size, y + window_size))
                sub_image_tensor = test_transformer(sub_image).unsqueeze(0).to(device) 

                # 进行前向传播
                with torch.no_grad():
                    output = model(sub_image_tensor)
                    score = sigmoid(output[0][1].item()) 
                    heatmap[y:y + window_size, x:x + window_size] += score 

    
        heatmap_min = np.min(heatmap)
        heatmap_max = np.max(heatmap)
        if heatmap_max > heatmap_min:  
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)

    
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Heatmap for {filename}')
        plt.axis('off')
        plt.savefig(f'heatmap_{filename}.png')  
        plt.close()

     
        mask = heatmap > 0.8

   
        gray_overlay = Image.new('RGBA', (width, height), (128, 128, 128, 200))
        overlay_array = np.array(gray_overlay)

       
        original_image = np.array(test_image.convert('RGBA')) 
        mapped_image = np.where(mask[:, :, None], overlay_array, original_image) 


        mapped_image_pil = Image.fromarray(mapped_image)
        mapped_image_pil.save(f'mapped_image_{filename}.png')

print("Heatmaps and mapped images generated successfully.")
