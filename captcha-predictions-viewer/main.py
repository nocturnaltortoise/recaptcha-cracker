import matplotlib.pyplot as plt
import json
import skimage.io

with open('../guesses.json', 'r') as predictions_file:
    for line in predictions_file:
        json_predictions = json.loads(line)

captcha_images = skimage.io.imread_collection('datasets/captchas/an apartment building/**/*')

fig, axes = plt.subplots(3,3)

for i, image in enumerate(captcha_images):
    for j in len(captcha_images) / 3:
        axes[i,j].imshow(image)

plt.show()
