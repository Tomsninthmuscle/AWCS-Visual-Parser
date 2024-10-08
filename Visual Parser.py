import os
import numpy as np
from PIL import Image
import torch
from PIL import ImageDraw
from scipy.interpolate import interp2d
from torchvision import transforms
import matplotlib.pyplot as plt
from scipy import interpolate

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'




folder_path = 'Your folder_path'


window_size = NumberA
stride = NumberB
scores = []

model_path = 'model.pt'





model = torch.load(model_path)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
  
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        image_path = os.path.join(folder_path, filename)
        test_image = Image.open(image_path)

        test_transformer = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])
        test_image_tensor = test_transformer(test_image).unsqueeze(0)

     
        with torch.no_grad():
            detections = []

            for y in range(0, test_image.height - window_size + 1, stride):
                for x in range(0, test_image.width - window_size + 1, stride):

               
                    window = test_image.crop((x, y, x + window_size, y + window_size))
                  
                    test_image_tensor = test_transformer(window).unsqueeze(0).to('cuda')
               
                    pred = model(test_image_tensor)
                    pred_label = torch.argmax(pred, dim=1).item()
                    if pred_label == 1:
                        detections.append((x, y, x + window_size, y + window_size))
                        scores.append(sigmoid(pred[0][1].item()))
                    else:
                        scores.append(1 - sigmoid(pred[0][0].item()))


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')


max_len = min(100, len(scores))
sqrt_len = int(np.sqrt(max_len))
scores_resized = np.resize(scores, (sqrt_len, sqrt_len))

x = list(range(sqrt_len))
y = list(range(sqrt_len))
X, Y = np.meshgrid(x, y)
Z = scores_resized


f = interpolate.interp2d(x, y, Z, kind='cubic')

x_new = np.linspace(min(x), max(x), 10)
y_new = np.linspace(min(y), max(y), 10)
Z_new = f(x_new, y_new)

X_new, Y_new = np.meshgrid(x_new, y_new)

ax.plot_surface(X_new, Y_new, Z_new, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

plt.show()

threshold = 0.8
projection_mask = Z_new > threshold


projection_mask = np.rot90(projection_mask)
projection_mask = np.rot90(projection_mask)
projection_mask = np.rot90(projection_mask)
projection_mask = np.fliplr(projection_mask)

projection_image = Image.new('RGB', test_image.size, (255, 255, 255))
projection_draw = ImageDraw.Draw(projection_image)


for i in range(sqrt_len):
    for j in range(sqrt_len):
        if projection_mask[i, j]:
            x_start = i * stride
            y_start = j * stride
            x_end = x_start + window_size
            y_end = y_start + window_size
            projection_draw.rectangle([x_start, y_start, x_end, y_end], fill=(255, 0, 0))


projection_image = projection_image.resize(test_image.size)

result_image = Image.blend(test_image.convert('RGBA'), projection_image.convert('RGBA'), alpha=0.3)

result_image.show()


fig, axes = plt.subplots(1, 2, figsize=(10, 5))


axes[0].imshow(test_image, cmap='gray')
axes[0].set_title('Original Image')


axes[1].imshow(Z_new, cmap='viridis', extent=[0, sqrt_len - 1, sqrt_len - 1, 0])
axes[1].set_title('Smooth Surface Height Map')

plt.show()
