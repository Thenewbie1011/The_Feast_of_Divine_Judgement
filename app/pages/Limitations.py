import streamlit as st
import base64

with open("Streamlit background4.png","rb") as image_file:
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
    .st-key-limitations_panel h2,
    .st-key-limitations_panel h3{{
    text-align: center;
    }}
    .st-key-limitations_panel{{
        background-color:rgba(0, 0, 0, 0.75);
        border-radius:10px;
        padding:10px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Limitations of Divine Judgment")
with st.container(border=True,key="limitations_panel"):
     st.header("A Word of Caution, Traveller!")
     st.write("Even mechanisms devised to aid the Seven have their limitations. While this system can identify food images and retrieve relevant " \
     "information, its verdict is not immune to mistakes. Keep the following limitations in mind before presenting a dish for judgement ")
     st.divider()
     st.header("I. The Eleven Culinary Categories")
     st.write("The mechanism recognizes only the 11 categories on which it was trained: Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles-Pasta, " \
     "Rice, Seafood, Soup and Vegetable-Fruit. ")
     st.write("A dish outside these categories may still be assigned to one of them. The system cannot reliably identify individual recipes, specific dishes " \
     "or ingredients")
     st.divider()
     st.header("II. The Quality of the Offering")
     st.write("The accuracy of the verdict depends partly on the image presented. Blurry photographs, poor lighting, unusual viewing angles, " \
     "obstructed dishes and cluttered backgrounds might affect prediction. ")
     st.write("For better results, present a clear, well lit image in which the food is easily visible")
     st.divider()
     st.header("III. The Verdict Is Not Absolute")
     st.write("The mechanism assigns probabilities to the eleven categories and selects the category with the highest predicted probability. " \
     "This does not guarantee that the selected category is correct.")
     st.write("The displayed probabilities represent the model's predictions, not a guarantee of correctness. The system may also struggle with unfamiliar " \
     "dishes or images that differ significantly from its training data.")
     st.divider()
     st.header("IV. The Divine Archives Have Their Limits")
     st.write("The nutritional information retrieved from MongoDB is associated with the predicted food category and in some cases, just one of the " \
     "categories such as in Noodles-Pasta, only one class' nutrition is present. It may not accurately represent the exact dish in the image, " \
     "its portion size, its recipe or its method of preparation. ")
     st.write("The system does not calculate the actual calories or nutrients present in the uploaded image. Its information should therefore be " \
     "treated as a general reference. ")
     st.divider()
     st.header("V. No Knowledge of Hidden Ingredients")
     st.write("The mechanism identifies visual food categories. It does not verify ingredients, detect allergens, assess food safety or determine " \
     "whether a dish is suitable for a particular dietary requirement")
     st.write("Do not rely on its predictions or nutritional records for allergy decisions, medical advice or food safety judgements")
     st.divider()
     st.header("VI. A Mechanism Shaped by Its Training")
     st.write("The model was trained using the Food-11 dataset. Its performance depends on the examples and variations represented in that dataset " \
     "Differences in cuisine, presentation, lighting and image quality may affect how well it generalises to new images.")
     st.write("During training, the model learns visual patterns that help it distinguish between food categories. However, it may also learn " \
     "patterns that are overly specific to the training images. If it relies too heavily on these patterns, it may misclassify unfamiliar " \
     "images or images that differ from those it encountered during training. ")
     st.write("Consequently, even when a dish appears similar to examples in the training dataset, the model's prediction may not always be correct. " \
     "Its learned patterns do not guarantee that it will recognise every food image accurately. ")
     st.divider()
     st.header("A Final Reminder, Traveller")
     st.write("The Feast of Divine Judgment is an educational AI project designed to demonstrate image classification and food information retrieval. " \
     "Its predictions are useful estimates, not unquestionable truths. Even the Seven must allow room for error!")
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