'''Training the MobileNetV2 model, hyperparameters would be supplied to this after hyperparamater tuning'''
import os
import random
import numpy as np
import tensorflow as tf
gpus=tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0],True)
tf.config.experimental.enable_op_determinism()
from preprocess import prepare_datasets
from model import Food11Model
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

SEED=42
os.environ["PYTHONHASHSEED"]=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT=Path(__file__).resolve().parent.parent
TRAINING_FIGURES_DIR=(PROJECT_ROOT/"outputs"/"figures"/"training_history")
TRAINING_FIGURES_DIR.mkdir(parents=True,exist_ok=True)

def train_model(hyperparameters):
    tf.keras.utils.set_random_seed(SEED)
    learning_rate=hyperparameters["learning_rate"]
    batch_size=hyperparameters["batch_size"]
    epochs=hyperparameters["epochs"]
    dropout_rate=hyperparameters["dropout_rate"]
    l2_rate=hyperparameters["l2_rate"]
    fine_tune_layers=hyperparameters["fine_tune_layers"]
    train_ds,validation_ds,test_ds=prepare_datasets(batch_size)
    food_11_model=Food11Model(dropout_rate=dropout_rate,l2_rate=l2_rate,fine_tune_layers=fine_tune_layers)
    model=food_11_model.get_model()
    #We would be using the Adam optimizer, NOT the SGD with momentum as has been suggested in a few papers
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate)
    '''We use the sparse categorical cross entropy here. We don't use categorical cross entropy as its meant when the target labels
    are one hot encoded values, sparse categorical cross entropy on the other hand is used when the target labels are integers'''
    model.compile(optimizer=optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"])
    #Early stop callback if no improvement is found
    early_stop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)
    history=model.fit(train_ds,validation_data=validation_ds,epochs=epochs,callbacks=[early_stop])
    return model,history,validation_ds,test_ds
def plot_training_history(history):
    #Plotting the training and validation accuracy
    plt.figure(figsize=(12,8))
    plt.plot(history.history["accuracy"],marker="o",linestyle="--",label="Training accuracy",color="green")
    plt.plot(history.history["val_accuracy"],marker='o',linestyle="--",color="blue",label="Validation accuracy")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Epoch vs training accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(TRAINING_FIGURES_DIR/"training_validation_accuracy.png",dpi=300,bbox_inches="tight")
    plt.show()
    #Plotting the training and validation loss
    plt.figure(figsize=(12,8))
    plt.plot(history.history["loss"],marker="o",linestyle="--",label="Training loss",color="green")
    plt.plot(history.history["val_loss"],marker='o',linestyle="--",color="blue",label="Validation loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Epoch vs training loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(TRAINING_FIGURES_DIR/"training_validation_loss.png",dpi=300,bbox_inches="tight")
    plt.show()