import tensorflow as tf
from data_loader import loadfood_11,CLASS_NAMES

'''This is as per the results of the paper:
Computer Vision in the Food Industry: Accurate, Real-time, and Automatic Food Recognition with Pretrained MobileNetV2
By Shayan Rokhva, Babak Teimourpour and Amir Hossein Soltani. 
The paper found that working with 256 x 256 images produced the best result on this particular dataset with the MobileNetV2 framework'''
IMAGE_SIZE=(256,256)
SEED=42
#Data augmentation pipeline
'''Data augmentation is used when we have minority classes and we need to synthesize more data for those classes.AU
 1. RandomFlip - Flips the image horizontally. The probability of a flip is 50%
 2. RandomRotation - Rotates the image. The fraction limits would be calculated with regards to a full rotation - 360 degrees or 2pi radians
  0.1 means a rotation range between -36 degrees and +36 degrees
 3. RandomBrightness - randomly changes the brightness by +/- fraction, which in this case, is 10%
 4. RandomContrast - randomly changes the contrast by +/- fraction, which in this case, is 10%
 5. RandomZoom - Positive values mean zooming in, while negative values mean zooming out
 6. RandomErasing - Randomly erases patches of an image and fills them up with random pixels. This has a 10% chance of happening that any
  given image in a batch undergoes this '''
data_augmentation=tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal",seed=SEED),
    tf.keras.layers.RandomRotation(0.1,seed=SEED),
    tf.keras.layers.RandomBrightness(0.1,seed=SEED),
    tf.keras.layers.RandomContrast(0.1,seed=SEED),
    tf.keras.layers.RandomZoom(0.1,seed=SEED),
    tf.keras.layers.RandomErasing(0.1,seed=SEED),
])
#For preprocessing a single image
def preprocess_img(image,training=False):
    #Resizing the image to fit the paper's results
    image=tf.image.resize(image,IMAGE_SIZE)
    '''VVIP - We ONLY augment the training dataset, NOT the validation or testing set'''
    if training:
        image=data_augmentation(image,training=True)
    '''This is performing standardization. This scales the image's raw pixel values to fit the requirement specified by mobilenetv2.
    This takes standard pixel values, ranging from [0,255] and scales them to [-1,1]'''
    #image=tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image
'''This is a very important function. We need to convert the dataframe into something that can be fed into the neural network. 
Now the main purpose of this function is to convert the image paths and classes produced by the load function into processed image tensors
and push them into batches. Now, a tensor is a way of storing numbers in organized structures of one or more dimensions. For example, a single number 
can be thought of a 0D tensor, a list of numbers as a 1D tensor, a table of numbers as a 2D tensor and several tables stacked together as a 3D tensor. 
Now for an colored image (RBG), the images consists of a grid called pixels and each pixel will have a certain value of R, B and G. Because of this,
images are represented as 3D tensors. While talking about batches, such as, (32,256,256,3) means that 32 represents the number of images and
each image has height of 256, width of 256 and has 3 channels. As neural networks work with numbers, converting the data obtained from the load
MUST be converted into tensors so that it can be fed into the neural network'''
def make_dataset(df,batch_size,training=False):
    #Extracts the image paths
    image_paths=df["image_path"].astype("str").values
    #Produces an integer label for each class
    labels=[]
    for class_name in df["class_name"]:
        labels.append(CLASS_NAMES.index(class_name))
    #Creates a tensorflow dataset object by slicing the arrays/lists pair by pair row-wise. It essentially groups each image path with its
    #appropriate label
    '''The main purpose of this function, without the inputs, is to create a dataset from in memory arrays or tensors by splitting along the
    first dimension. The output itself is a tensorflow object'''
    dataset=tf.data.Dataset.from_tensor_slices((image_paths,labels))
    #Shuffling the training images
    if training:
        dataset=dataset.shuffle(len(df),seed=SEED,reshuffle_each_iteration=True)
    def load_image(path,label):
        #Reads the raw file bytes of the image, it is not yet a usable image tensor
        image=tf.io.read_file(path)
        #The raw file bytes are now converted into an image tensor
        image=tf.io.decode_image(image,channels=3,expand_animations=False)
        image=preprocess_img(image,training)
        return image,label
    '''This is where everything comes together for this function. Before the load_image function, we had the path and the corresponding label. 
    The load_function reads the raw file bytes, converts them into an image tensor, applies preprocessing techniques (depending on the set) and
    returns the image along with the label. After mapping, we get a mapping between the image tensors and the label, so everything is numbers here
    We also enable parallel processing for images in order to improve speed'''
    dataset=dataset.map(load_image,num_parallel_calls=1 if training else 4,deterministic=True)
    #Groups individual items into batches of size 32
    dataset=dataset.batch(batch_size)
    #Prepares the next batch in the background
    dataset=dataset.prefetch(2)
    return dataset
def prepare_datasets(batch_size):
    train_df,validation_df,test_df=loadfood_11()
    return(
        make_dataset(train_df,batch_size,training=True),
        make_dataset(validation_df,batch_size),
        make_dataset(test_df,batch_size)
    )
if __name__=="__main__":
    train_ds,validation_ds,test_ds=prepare_datasets(batch_size=32)
    #Retrieving the first batch of 32 images and their labels
    images,labels=next(iter(train_ds))
    print("Preprocessing setup successful.")
    print(f"Batch shape: {images.shape}")
    print(f"Label shape: {labels.shape}")