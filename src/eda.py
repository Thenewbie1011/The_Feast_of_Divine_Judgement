import matplotlib.pyplot as plt
import cv2
from data_loader import loadfood_11,getClassDistribution
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parent.parent
OUTPUT_DIR=PROJECT_ROOT/"outputs"/"figures"
CLASS_DISTRIBUTION_DIR=OUTPUT_DIR/"class_distributions"
SAMPLE_IMGS_DIR=OUTPUT_DIR/"sample_images_from_each_class"
CLASS_DISTRIBUTION_DIR.mkdir(parents=True,exist_ok=True)
SAMPLE_IMGS_DIR.mkdir(parents=True,exist_ok=True)
train_df,validation_df,test_df=loadfood_11()
train_counts=getClassDistribution(train_df)
valid_counts=getClassDistribution(validation_df)
test_counts=getClassDistribution(test_df)
#Plotting the training set class distribution
plt.figure(figsize=(12,8))
plt.bar(
    train_counts.index,
    train_counts.values
)
plt.xlabel("Food 11 class")
plt.ylabel("Number of images")
plt.title("Training set class distribution")
plt.tight_layout()
plt.savefig(CLASS_DISTRIBUTION_DIR/"training_set_class_distribution.png",dpi=300,bbox_inches="tight")
plt.show()
#Plotting the validation set class distribution
plt.figure(figsize=(12,8))
plt.bar(
    valid_counts.index,
    valid_counts.values
)
plt.xlabel("Food 11 classes")
plt.ylabel("Number of images")
plt.title("Validation set class distribution")
plt.tight_layout()
plt.savefig(CLASS_DISTRIBUTION_DIR/"validation_set_class_distribution.png",dpi=300,bbox_inches="tight")
plt.show()
#Plotting the test set class distribution
plt.figure(figsize=(12,8))
plt.bar(
    test_counts.index,
    test_counts.values
)
plt.xlabel("Food 11 classes")
plt.ylabel("Number of images")
plt.title("Test set class distribution")
plt.tight_layout()
plt.savefig(CLASS_DISTRIBUTION_DIR/"test_set_class_distribution.png",dpi=300,bbox_inches="tight")
plt.show()
#Displaying a sample food item from each class (using the training set)
unique_class_names=train_df["class_name"].unique()
plt.figure(figsize=(12,10))
for i,class_name in enumerate(unique_class_names):
    class_images=train_df[train_df["class_name"]==class_name]
    image_path=class_images.iloc[0]["image_path"]
    #Using OpenCV to view the image
    image=cv2.imread(str(image_path))
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    plt.subplot(3,4,i+1)
    plt.imshow(image)
    plt.title(class_name)
    plt.axis("off")
plt.tight_layout()
plt.savefig(SAMPLE_IMGS_DIR/"img_samples.png",dpi=300,bbox_inches="tight")
plt.show()