import streamlit as st
import tensorflow as tf
import numpy as np
#Used in order to open the image
from PIL import Image
import base64
with open("Streamlit background4.png","rb") as image_file:
     encoded_image=base64.b64encode(image_file.read()).decode()
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image:url("data:image/png;base64,{encoded_image}");
        background-size:cover;
        background-position:center;
        background-attachment:fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
    #Styling both the left, right, the class probabilities and the header to be a bit more opaque so that the text is visible
    #0.75 indicates a high level of opaqueness but not entirely opaque
    #We also make it seem that the left and right panels appear as one
st.markdown(
    """
    <style>
    /* Main panel (targets the container created with key="main_panel") */
    .st-key-main_panel {
        background-color: rgba(0, 0, 0, 0.75) !important;
        border-radius: 10px;
        padding: 10px;
    }
    /* Remove gap between the two columns */
    div[data-testid="stHorizontalBlock"]{
        gap:0px;
    }
    /* Columns */
    div[data-testid="stColumn"]{
        padding:20px;
        box-sizing:border-box;
    }
    /* Title */
    h1{
        background-color:rgba(0,0,0,0.75);
        padding:15px;
        border-radius:10px;
        text-align:center;
    }
    /* Class probability expander */
    div[data-testid="stExpander"]{
        background-color:transparent;
        border-radius:10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
from database import foods_collection
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
MODEL_PATH="../outputs/saved_model/food11_mobilenetv2.keras"
model=tf.keras.models.load_model(MODEL_PATH)
st.title("The Feast of Divine Judgment")
with st.container(key="main_panel"):
    st.markdown(
    "<h3 style='text-align:center;'>Upload your culinary invention to be judged by the 7 archons of Tevyat!</h3>",
    unsafe_allow_html=True
    )
    uploaded_file=st.file_uploader("",type=["jpg","jpeg","png","bmp","gif","webp"],accept_multiple_files=False)
    if(uploaded_file is not None):
        #Converts the uploaded image to RGB
        image=Image.open(uploaded_file).convert("RGB")
        #Organizing the UI into two columns - one for uploading and one for result prediction
        left_column,right_column=st.columns(2)
        with left_column:
            st.subheader("Uploaded image")
            st.image(uploaded_file,caption="Uploaded image",width="stretch")
        #Converts to a numpy array
        image_array=np.array(image)
        #Resizes to 256 x 256
        image_array=tf.image.resize(image_array,(256, 256))
        #Introduces the batch dimension as neural networks expect a batch of images so this becomes (1,256,256,3)
        image_array=tf.expand_dims(image_array,axis=0)
        predictions=model.predict(image_array,verbose=0)
        #Finds the class with the highest probability
        predicted_index=np.argmax(predictions[0])
        predicted_class=CLASS_NAMES[predicted_index]
        #Probability of the predicted class
        confidence=predictions[0][predicted_index]*100
        with right_column:
            st.subheader("Archons' Verdict")
            st.write(f"**Food:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.subheader("Information from the Divine Archives of Sumeru")
        #This is an important one. Python has this way of querying MongoDB to retrieve a particular document for a particular specification. You
        #use find_one() to this end. So right now, we are finding the document containing information about the predicted class using the filtering
        #criteria of class name. This function is available using the PyMongo library'''
            food=foods_collection.find_one({"class_name":predicted_class})
            if(food):
                st.write(f"**Description:** {food['description']}")
                st.write(f"**Calories:** {food['calories']} kcal")
                st.write(f"**Protein:** {food['protein']} g")
                st.write(f"**Carbohydrates:** {food['carbohydrates']} g")
                st.write(f"**Link to the nutritional information:** {food['link']}")
        #Class probabilities are optional now for viewing        
        with st.expander("Class Probabilities"):
            for class_name, probability in zip(CLASS_NAMES, predictions[0]):
                st.write(f"**{class_name}:** {probability * 100:.2f}%")
                st.progress(float(probability))
    st.markdown(
        """
        <div style="
            background-color: transparent;
            border: none;
            outline: none;
            box-shadow: none;
            padding: 12px;
            text-align: center;
            color: white;
            margin-top: 15px;
        ">
            <b>An AI-powered food classification project inspired by Genshin Impact.</b>
            <br>
            <b>DISCLAIMER: The model might not always be right in its classification; take its predictions with a pinch of salt.</b>
            <br>
            <b>Made by: Shreyas Nitin</b>
        </div>
        """,
        unsafe_allow_html=True
    )