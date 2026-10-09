import streamlit as st
import base64
from pathlib import Path
BACKGROUND_PATH=Path(__file__).resolve().parent.parent/"Streamlit background4.png"
with open(BACKGROUND_PATH,"rb") as image_file:
     encoded_image=base64.b64encode(image_file.read()).decode()
st.markdown(
    f"""
    <style>
    .stApp{{
        background-image:url("data:image/png;base64,{encoded_image}");
        background-size:cover;
        background-position:center;
        background-attachment:fixed;
    }}
    h1{{
        background-color:rgba(0, 0, 0, 0.75);
        padding:15px;
        border-radius:10px;
        text-align:center;
    }}
    .st-key-about_panel h2,
    .st-key-about_panel h3{{
    text-align: center;
    }}
    .st-key-about_panel{{
        background-color:rgba(0, 0, 0, 0.75);
        border-radius:10px;
        padding:10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("About The Feast of Divine Judgment")
with st.container(border=True, key="about_panel"):
     st.header("Welcome Traveller!")
     st.write("After the battle against Ronova, the Shade of Death, the 7 archons of Tevyat gather for a round table to discuss the next steps " \
     "against the Heavenly Principles and dealing with the Celestial Nails hanging above their nations, with destruction looming. However, in a brief " \
     "moment of respite, the 7 archons turn to you, their trusted friend, the one who has traversed the 7 nations, in order to come up with dishes that " \
     "they can taste and identify. ")
     st.divider()
     st.header("The Rules of Judgement")
     st.write("The rules of this gathering are simple. Present a dish before the 7 archons and let their verdict be delivered. Though each Archon has " \
     "their own perspective and ideal, the judgement must be impartial. Thus, the archons have relied on an ancient mechanism founded by Nibelung, " \
     "the second descender, in order to make the judgement. ")
     st.write("This mechanism is a neural network powered by thousands of culinary images. Powered by MobileNetV2, it analyzes the image and predicts " \
     "the most likely category. The category with the highest probability becomes the **archons' verdict**. The probabilities for all **eleven classes** " \
     "can also be examined, offering a glimpse into the mechanism's reasoning. ")
     st.divider()
     st.header("The Eleven Culinary Categories")
     st.write("The mechanism created by Nibelung is fairly nascent. Hence, it can only provide a judgement on a few classes, which are listed below: ")
     st.write("""
    - Bread
    - Dairy product
    - Dessert
    - Egg
    - Fried food
    - Meat
    - Noodles-Pasta
    - Rice
    - Seafood
    - Soup
    - Vegetable-Fruit
    """)
     st.divider()
     st.header("The Divine Archives of Sumeru")
     st.write("Identifying the dish is only the beginning. The Divine Archives, stored in the depths of the Aaru system, is maintained by MongoDB. " \
     "Once a verdict has been reached, Aaru reveals the records showing the description nutritional information about the class of the dish identified. " \
     "With the verdict delivered and the relevant records revealed, even the Gods, who survived the deluge of the Cataclysm, might discover something new " \
     "about the food placed before them. ")
     st.divider()
     st.subheader("The Powers Behind the Judgement")
     st.markdown(
        """
        - **MobileNetV2:** The model architecture responsible for classifying dishes.
        - **TensorFlow and Keras:** The frameworks used to build and run the model.
        - **Food-11:** The dataset used to train and evaluate the classifier.
        - **MongoDB:** The Divine Archives storing food descriptions and nutritional information.
        - **Streamlit:** The interface through which you present your culinary creations.
        """
    )
     st.divider()
     st.markdown(
    """
    <div style="text-align:center;">
        Now then, Traveller, what dish shall we present before the 7 Archons?
    </div>
    """,
    unsafe_allow_html=True
    )
     st.divider()
     st.markdown(
        """
        <div style="
            text-align: center;
            color: white;
            padding: 12px;
            margin-top: 15px;
        ">
            <b>An AI-powered food classification project inspired by Genshin Impact.</b>
            <br>
            <b>Made by: Shreyas Nitin</b>
        </div>
        """,
        unsafe_allow_html=True
    )