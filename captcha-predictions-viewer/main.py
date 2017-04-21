import matplotlib.pyplot as plt
import json
import skimage.io

with open('../guesses.json', 'r') as predictions_file:
    predictions = predictions_file.read()
    json_predictions = json.loads(predictions)
    apartment_predictions = json_predictions['an apartment building']



for subfolder_name in apartment_predictions:
    captcha_images = skimage.io.imread_collection('../datasets/captchas/an apartment building/{0}/*'.format(subfolder_name))
    fig, axes = plt.subplots(3,3)
    plt.rc('font', size=8)

    subfolder_contents = apartment_predictions[subfolder_name]
    predictions = subfolder_contents['predictions']
    matching_predictions = subfolder_contents['matching_predictions']
    for i in range(3):
        for j in range(3):
            image = captcha_images[i * 3 + j]
            prediction = predictions[i * 3 + j]
            axes[i, j].imshow(image)
            if prediction in matching_predictions:
                axes[i, j].set_title("Picked\n {0}".format(prediction))
            else:
                axes[i, j].set_title(prediction)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.subplots_adjust(hspace=0.5)
    plt.show()
