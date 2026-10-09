from pathlib import Path
import pandas as pd
'''We are getting the project root from this. __file__ represents the current python file path. As we are in data_loader.py,we
get Food11/src/data_loader.py. The resolve() turns it into an absolute path and parent.parent moves the path up two levels. We thus get the absolute
path to Food11 from the drive itself'''
PROJECT_ROOT=Path(__file__).resolve().parent.parent
#Path to the food 11 dataset
#DATASET_DIR=PROJECT_ROOT/"data"/"raw"/"Food_11"
DATASET_DIR=Path("/home/shreyas/Food11_data/Food_11")
#Creating the paths to the individual splits
TRAIN_DIR=DATASET_DIR/"training"
VALIDATION_DIR=DATASET_DIR/"validation"
TEST_DIR=DATASET_DIR/"evaluation"
#11 classes
CLASS_NAMES=[
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit"
]
IMAGE_EXTENSIONS={
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".webp"
}
#This function is needed in order to retrieve the images and their classes for a particular folder. This function takes in the path to that folder
def load_split(split_dir):
    #Converts the input into a path
    split_dir=Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset split not found {split_dir}")
    #Stores information about every image
    records=[]
    '''This is where things start getting interesting. As our goal is to identify the class for a particular image, we would need
    to go inside the directory. This is done in the following steps. The first step is to get the class inside of our chosen folder. There
    are 11 classes and hence 11 folders for each directory. This is what iterdir does. It gets the immediate subfolders/files for a given folder.
    However, it doesn't operate in a nested way, so we would need to have 2 steps. The first is getting the folder for the class and then going inside
    that folder, again using iterdir'''
    for class_dir in sorted(split_dir.iterdir()):
        #Checks if the class is a folder or not
        if not class_dir.is_dir():
            continue
        #Gets the name of the folder
        class_name=class_dir.name
        '''This is the second stage of the process. Now that we are in the folder, we go and check the images, again using iterdir'''
        for img_path in sorted(class_dir.iterdir()):
            #Checks if the image is not a file
            if not img_path.is_file():
                continue
            #Checks if the image extension is not in the list of available extensions
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            #Appends the full path of the image along with its class to the records folder
            records.append(
                {
                    "image_path":img_path,
                    "class_name":class_name
                }
            )
    return pd.DataFrame(records)
#Loads the complete food 11 dataset
def loadfood_11():
    train_df=load_split(TRAIN_DIR)
    valid_df=load_split(VALIDATION_DIR)
    test_df=load_split(TEST_DIR)
    return train_df, valid_df,test_df
#Verifying that everything has been loaded properly
def verify_classes(train_df,valid_df,test_df):
    train_classes=set(train_df["class_name"])
    validation_classes=set(valid_df["class_name"])
    test_classes=set(test_df["class_name"])
    expected_classes=set(CLASS_NAMES)
    if(train_classes!=expected_classes):
        raise ValueError("The training set does not contain the expected number of classes")
    if(validation_classes!=expected_classes):
        raise ValueError("The validation set does not contain the expected number of classes")
    if(test_classes!=expected_classes):
        raise ValueError("The test set does not contain the expected number of classes")
    return CLASS_NAMES.copy()
def getClassDistribution(df):
    return (df["class_name"].value_counts().reindex(CLASS_NAMES,fill_value=0))
if __name__=="__main__":
    train_df,validation_df,test_df=loadfood_11()
    class_names=verify_classes(train_df,validation_df,test_df)
    print("Food 11 dataset loaded successfully!")
    print(f"Number of training images: {len(train_df)}")
    print(f"Number of validation images: {len(validation_df)}")
    print(f"Number of testing images: {len(test_df)}")
    print("\n Classes: \n")
    for class_name in class_names:
        print(f" - {class_name}")
    print("Training class distribution: \n")
    print(getClassDistribution(train_df))
    print("Validation class distribution: \n")
    print(getClassDistribution(validation_df))
    print("Test class distribution: \n")
    print(getClassDistribution(test_df))