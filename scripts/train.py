import numpy as np
from modules import (tree_dataset,
    model_mobilenet,
    model_mobilenet_2_aug,
    model_mobilenet_3_bigger_dec,
    model_mobilenet_4_smaller_dec
)
import tensorflow as tf
from matplotlib import pyplot as plt
from pathlib import Path


def display(display_list, save):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')

    if save:
        plt.savefig(results_path + "prediction_example.png")

    plt.show()



def create_mask(pred_mask):
    pred_mask = tf.sigmoid(pred_mask)
    pred_mask = pred_mask > 0.5
    return tf.cast(pred_mask[0], tf.uint8)


def show_predictions(num=1, save=False):
    for image, mask in train_ds.take(num):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)], save)


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


# TRAINING

train_ds, val_ds, test_ds = tree_dataset.assemble()
model, model_label = model_mobilenet_4_smaller_dec.build()
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

# show_predictions()

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=0.0,
    restore_best_weights=True
)

model_history = model.fit(
    train_ds,
    epochs=500,
    validation_data=val_ds,
    callbacks=[DisplayCallback(), early_stop]
)

# SAVING RESULTS

results_path = "../results/" + model_label
Path(results_path).mkdir(exist_ok=True)
results_path = results_path + "/"

model.save(results_path + "model.keras")


loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig(results_path + "training_history.png")
plt.show()

best_epoch = np.argmin(val_loss) + 1
final_train_loss = loss[best_epoch]
final_train_acc  = model_history.history["accuracy"][best_epoch]
final_val_loss = val_loss[best_epoch]
final_val_acc  = model_history.history["val_accuracy"][best_epoch]
with open(results_path + "metrics.txt", "w") as f:
    f.write(f"Final training loss: {final_train_loss:.3f}\n")
    f.write(f"Final training accuracy: {final_train_acc:.3f}\n")
    f.write(f"Final validation loss: {final_val_loss:.3f}\n")
    f.write(f"Final validation accuracy: {final_val_acc:.3f}\n")

show_predictions(save=True)