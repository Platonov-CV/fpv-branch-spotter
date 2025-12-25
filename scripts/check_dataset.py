from modules import tree_dataset
from matplotlib import pyplot as plt
import tensorflow as tf


train_ds, val_ds, test_ds = tree_dataset.assemble()

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

for images, masks in train_ds.take(2):
  sample_image, sample_mask = images[0], masks[0]
  display([sample_image, sample_mask])
