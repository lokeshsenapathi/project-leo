import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


st.set_page_config(page_title="AI RetinaVision", page_icon="icons/braille-solid.svg", initial_sidebar_state='collapsed')

# CSS style for initializing model
css_model_init = '''
<style>
.model-init {
    font-size: 15px;
    color: #888888;
}
</style>
'''

# CSS style for footer
css_copyr = '''
<style>
.footer-text {
    font-size: 15px;
    color: #888888;
    text-align: center;
}
</style>
'''

# CSS style for Font Awesome icons
css_fa = '''                                                                                                                                                     
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css,%s">
<style>
.footer-fa {
    font-size: 20px;
    color: purple;
    color: yellow;
    text-align: center;
    margin: 0 5px;
    display: inline-block;
}
.footer-icons {
    text-align: center;
    background-image: url("C:/Users/senapathi vinni/Desktop/pythonProject11/OCT-Retinal-Disease-Detection-CNN/fotor-ai-20240318212942.jpg");
    background-size: cover;
    padding: 20px; /* Adjust padding as needed */
}
</style>
'''


# Inject the CSS into the Streamlit app using st.markdown

# Apply CSS styles
st.markdown(css_model_init, unsafe_allow_html=True)
st.markdown(css_copyr, unsafe_allow_html=True)
st.markdown(css_fa, unsafe_allow_html=True)






# Main content
st.markdown('<h1 style="text-align: center;"><i class="fa-solid fa-braille"></i> &nbspRetinaVision AI </h1>',
            unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'><i>Retinal tissue abnormality detection using AI</i></h2>",
            unsafe_allow_html=True)

st.markdown("""
# Welcome to RetinaVision AI

Unleash the power of AI in healthcare with RetinaVision AI! Our intuitive web app lets you upload retinal optical coherence tomography (OCT) images and get instant predictions from our cutting-edge model. Whether you're a healthcare professional or just curious, explore how deep learning detects abnormalities in retinal tissue. Join us in revolutionizing eye health through innovative technology. Upload your image now and witness the future of vision.
""")

# Initializing the model
st.markdown('<p class="model-init">Initializing the model...</p>', unsafe_allow_html=True)
densenet121 = torchvision.models.densenet121()
densenet121.classifier = nn.Linear(1024, 2)

# Loading weights
st.markdown('<p class="model-init">Loading weights...</p>', unsafe_allow_html=True)
densenet121.load_state_dict(torch.load('models/densenet121-oct-5metrics-v1.pt', map_location=torch.device('cpu')))
densenet121.eval()

# Model ready
st.markdown('<p class="model-init">Model is ready ✔</p>', unsafe_allow_html=True)

# File upload
images = st.file_uploader("**CHOOSE AN IMAGE**", type=['jpg'], accept_multiple_files=True)

# Predictions
if images is not None:
    filenames = [image.name for image in images]
    transforms = T.Compose([
        T.Resize(256, interpolation=InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensors = []
    for image in images:
        img = Image.open(image)
        if len(images) == 1:
            st.image(img, use_column_width=True, caption=image.name)
        if img.getbands() != 'RGB':
            img = img.convert('RGB')
        transformed_img = transforms(img)
        img_tensors.append(transformed_img)
    inp_tensor = None
    if len(img_tensors) > 0:
        inp_tensor = torch.stack(img_tensors)

    if len(images) > 0:
        def visualize_file(file_name: str):
            if file_name is not "":
                if file_name in filenames:
                    for img in images:
                        if img.name == vis_filename:
                            vis_img = Image.open(img)
                            return st.image(vis_img, use_column_width=True, caption=vis_filename)
                return st.warning(
                    f':orange["`{vis_filename}`" not found in your uploads! Please recheck the filename.]')


        if len(images) > 1:
            st.markdown("**Want to visualize a specific image file?**")
            vis_filename = st.text_input(label="Specify the filename", placeholder="my_retinal_oct_image.jpeg")
            visualize_file(str(vis_filename))

        st.markdown('')
        st.markdown('')

        if st.button("**Predict**",
                     help="The model evaluates all the given images and returns predicted classes and respective confidence scores."):
            def predict(tensor):
                raw = densenet121(tensor)
                y_hat = softmax(raw, dim=1)
                category = torch.argmax(y_hat, dim=1).tolist()
                return category, y_hat.tolist()


            predictions = []
            probability = []
            output = predict(inp_tensor)
            for cat, prob in zip(output[0], output[1]):
                if cat == 0:
                    predictions.append("ABNORMAL")
                    probability.append({round(float(prob[0]) * 100, 2)})
                elif cat == 1:
                    predictions.append("NORMAL")
                    probability.append({round(float(prob[1]) * 100, 2)})

            st.markdown('')
            st.markdown('')
            st.markdown('### Results')
            df = pd.DataFrame()
            df['Filename'] = filenames
            df['Predicted Category'] = predictions
            df['Confidence (%)'] = probability
            st.dataframe(df, use_container_width=True,
                         column_order=['Filename', 'Predicted Category', 'Confidence (%)'])
            st.markdown('')

st.markdown('')
st.markdown('')
st.markdown('')
st.markdown('***')

# Footer
st.markdown('<div class="footer-icons">', unsafe_allow_html=True)
st.markdown(
    '   <p>For more information visit <a href="http://www.lokeshsenapathi.website">www.lokeshsemnapathi.website</a></p>',
    unsafe_allow_html=True)
st.markdown('   Happy Coding', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
