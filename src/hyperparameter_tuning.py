'''Important design decision. Usually, randomizedsearchCV is used for hyperparameter tuning. However, its main feature is the k fold cross
validation. We have a dedicated validation set so using randomizedsearchCV would be pretty complicated here. We instead use ParameterSampler
which is basically like randomizedsearchCV but without the cross validation. It produces random sample combinations from the
given parameter distribution. We can then train the model using the given parameters, evaluate it on the validation set and keep a score
on how good it performed with that particular parameter set. '''
import os
import random
import tensorflow as tf
gpus=tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0],True)
tf.config.experimental.enable_op_determinism()
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterSampler
from preprocess import prepare_datasets
from model import Food11Model
import csv
import json
from pathlib import Path
import gc
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("mixed_float16")

SEED=42
os.environ["PYTHONHASHSEED"]=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
PROJECT_ROOT=Path(__file__).resolve().parent.parent
TUNING_LOG_DIR=PROJECT_ROOT/"outputs"/"tuning_logs"
TUNING_LOG_DIR.mkdir(parents=True,exist_ok=True)
TUNING_LOG_FILE=TUNING_LOG_DIR/"hyperparameter_tuning.csv"
BEST_PARAMETERS_FILE=PROJECT_ROOT/"outputs"/"best_hyperparameters.json"

parameter_distributions={
    "learning_rate":[1e-5, 5e-5, 1e-4, 5e-4],
    "batch_size":[16, 32],
    "epochs":[20,25,30],
    "dropout_rate":[0.1,0.2,0.3,0.4],
    "l2_rate":[0.00001,0.0001,0.001],
    "fine_tune_layers":[30, 60, 100]
}
#Generating a list of parameter combinations
parameter_combinations=list(ParameterSampler(parameter_distributions,n_iter=20,random_state=SEED))
def evaluate_on_validation(model,validation_ds):
    #y_true=[]
    #y_pred=[]
    #for images,labels in validation_ds:
        #predictions=model(images,training=False)
        #y_true.extend(labels.numpy())
        #y_pred.extend(np.argmax(predictions.numpy(),axis=1))
    y_true=np.concatenate([labels.numpy() for _,labels in validation_ds])
    y_pred=np.argmax(model.predict(validation_ds,verbose=0),axis=1)
    return f1_score(y_true,y_pred,average="macro")
def tune_hyperparameters():
    best_score=-1
    best_hyperparameters=None
    with open(TUNING_LOG_FILE,"w",newline="") as file:
        writer=csv.writer(file)
        writer.writerow([
            "trial",
            "learning_rate",
            "batch_size",
            "epochs",
            "dropout_rate",
            "l2_rate",
            "fine_tune_layers",
            "validation_macro_f1"
        ])
        for trial,parameters in enumerate(parameter_combinations,start=1):
            print(f"Trial: {trial}/20")
            print(parameters)
            #Clears garbage memory
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(SEED)
            train_ds,validation_ds,_=prepare_datasets(parameters["batch_size"])
            food_11_model=Food11Model(dropout_rate=parameters["dropout_rate"], l2_rate=parameters["l2_rate"],fine_tune_layers=parameters["fine_tune_layers"])
            model=food_11_model.get_model()
            optimizer=tf.keras.optimizers.Adam(learning_rate=parameters["learning_rate"])
            model.compile(optimizer=optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"])
            early_stop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)
            try:
                model.fit(train_ds,validation_data=validation_ds,epochs=parameters["epochs"],callbacks=[early_stop],verbose=1)
                macro_f1=evaluate_on_validation(model,validation_ds)
                print(f"Validation macro F1: {macro_f1}")
            except tf.errors.ResourceExhaustedError:
                print("GPU OOM - skipping this trial.")
                macro_f1=-1
            writer.writerow([
                trial,
                parameters["learning_rate"],
                parameters["batch_size"],
                parameters["epochs"],
                parameters["dropout_rate"],
                parameters["l2_rate"],
                parameters["fine_tune_layers"],
                macro_f1
            ])
            file.flush()
            if(macro_f1>best_score):
                best_score=macro_f1
                best_hyperparameters=parameters
            del model
            del food_11_model
            del train_ds
            del validation_ds
            del early_stop
            del optimizer
            tf.keras.backend.clear_session()
            gc.collect()
    with open(BEST_PARAMETERS_FILE,"w") as file:
        json.dump({
            "hyperparameters":best_hyperparameters,
            "validation_macro_f1":best_score
        },file,indent=4)
    return best_hyperparameters,best_score