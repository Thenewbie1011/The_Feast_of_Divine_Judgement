import os
import random
import numpy as np
import tensorflow as tf

SEED=42
os.environ["PYTHONHASHSEED"]=str(SEED)
os.environ["TF_DETERMINISTIC_OPS"]="1"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import(
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from hyperparameter_tuning import tune_hyperparameters
from train import train_model,plot_training_history
from data_loader import CLASS_NAMES
PROJECT_ROOT=Path(__file__).resolve().parent.parent
MODEL_DIR=PROJECT_ROOT/"outputs"/"saved_model"
TEST_RESULTS_DIR=PROJECT_ROOT/"outputs"/"test_results"
TEST_RESULTS_DIR.mkdir(parents=True,exist_ok=True)
MODEL_DIR.mkdir(parents=True,exist_ok=True)
best_hyperparameters,best_score=tune_hyperparameters()
print("Best hyperparamaters: ")
print(best_hyperparameters)
print(f"Best validation macro f1 score: {best_score}")
model,history,validation_ds,test_ds=train_model(best_hyperparameters)
model.save(MODEL_DIR/"food11_mobilenetv2.keras")
plot_training_history(history)
test_loss,test_accuracy=model.evaluate(test_ds,verbose=1)
print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")
y_true=[]
for images,labels in test_ds:
    for label in labels.numpy():
        y_true.append(label)
predictions=model.predict(test_ds,verbose=1)
y_pred=[]
for prediction in predictions:
    predicted_class=np.argmax(prediction)
    y_pred.append(predicted_class)
acc_score=accuracy_score(y_true,y_pred)
macro_precision=precision_score(y_true,y_pred,average="macro")
macro_recall=recall_score(y_true,y_pred,average="macro")
macro_f1=f1_score(y_true,y_pred,average="macro")
statistics_file=TEST_RESULTS_DIR/"test_statistics.txt"
with open(statistics_file,"w") as file:
    file.write("Food 11 dataset test statistics\n")
    file.write("---------------------------------")
    file.write(f"Test loss: {test_loss:.4f}\n")
    file.write(f"Test accuracy: {test_accuracy:.4f}\n")
    file.write(f"Test macro precision: {macro_precision:.4f}\n")
    file.write(f"Test macro recall: {macro_recall:.4f}\n")
    file.write(f"Test macro F1: {macro_f1:.4f}\n")
print("Test statistics: ")
print(f"Test macro precision: {macro_precision}")
print(f"Test macro recall: {macro_recall}")
print(f"Test macro F1: {macro_f1}")
print("Classification report: ")
print("-------------------------")
classification_report_text=classification_report(y_true,y_pred,target_names=CLASS_NAMES,digits=4)
classification_report_file=TEST_RESULTS_DIR/"classification_report.txt"
with open(classification_report_file, "w") as file:
    file.write("Food-11 Classification Report\n")
    file.write("-----------------------------\n")
    file.write(classification_report_text)
print(classification_report_text)
confusion=confusion_matrix(y_true,y_pred)
display=ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=CLASS_NAMES)
fig,ax=plt.subplots(figsize=(12,10))
display.plot(ax=ax)
plt.title("Food 11 dataset confusion matrix")
plt.tight_layout()
plt.savefig(TEST_RESULTS_DIR/"confusion_matrix.png",dpi=300,bbox_inches="tight")
plt.show()