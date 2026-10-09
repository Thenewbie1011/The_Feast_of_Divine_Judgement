'''We are using transfer learning using MobileNetV2. This is for just defining the model, NOT training or tuning it'''
import tensorflow as tf

IMAGE_SIZE=(256,256)
NUM_CLASSES=11

class Food11Model:
    def __init__(self,dropout_rate=0.2,l2_rate=0.0001,fine_tune_layers=30):
        self.base_model=tf.keras.applications.MobileNetV2(
                input_shape=IMAGE_SIZE+(3,),
                #This removes the classification layer since we don't need their classification layer
                include_top=False,
                #Loads the learnt weights from imagenet
                weights="imagenet"
            )
        #This doesn't modify the weights during training #Running an exp
        self.base_model.trainable=True
        for layer in self.base_model.layers[:-fine_tune_layers]:
            layer.trainable=False
        for layer in self.base_model.layers[-fine_tune_layers:]:
            if(isinstance(layer,tf.keras.layers.BatchNormalization)):
                layer.trainable=False
        self.model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=IMAGE_SIZE+(3,)),
        #Transforms pixel values from [0,255] to [-1,1]
        tf.keras.layers.Rescaling(1/127.5,offset=-1),
        self.base_model,
        #This is important as it is akin to flatten before it is fed into the dense layers
        #Furthermore, this basically finds whether a particular fearture is found strongly overall
        #An argument could be made for using flatten here but until the last layer, mobilenetv2 would not produce a number, let's say it
        #produces 7 x 7 x 1280. Flattening it would result in an 1D array of length 62720 and this would require a LOT more parameters in the
        #Final layers. globalaveragepooling2d takes the average of each spatial map and this results in only 1280 values being input into
        #The final layers
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(NUM_CLASSES,activation="softmax",dtype="float32",kernel_regularizer=tf.keras.regularizers.l2(l2_rate))
    ])
    def get_model(self):
        return self.model
if __name__=="__main__":
    model=Food11Model()
    model=model.get_model()
    model.summary()