import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def sigmoid(x):
    return 1 / (1 + np.exp(-x))



folder_path = 'Yourfolder_path'

window_size = 320
stride = 160



model.load_state_dict(torch.load('your_best_model.pt'))
model.eval()
model.to(device)

test_transformer = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)


        test_image = Image.open(image_path)
        width, height = test_image.size


        heatmap = np.zeros((height, width), dtype=np.float32)


        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
           
                sub_image = test_image.crop((x, y, x + window_size, y + window_size))
                sub_image_tensor = test_transformer(sub_image).unsqueeze(0).to(device)

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


        mask_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        mask_rgba[..., 0:3] = (128, 128, 128)
        mask_rgba[..., 3] = (mask * 100).astype(np.uint8)

        original_image_rgba = np.array(test_image.convert('RGBA'))
        mapped_image = Image.alpha_composite(Image.fromarray(original_image_rgba), Image.fromarray(mask_rgba))

 
        mapped_image.save(f'mapped_image_{filename}.png')

print("success!")
