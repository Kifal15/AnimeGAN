import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


st.title("GAN Image Generator")
st.write("Generate and display images using a pre-trained GAN model.")

model_path = "./generator.h5"  
try:
    model = tf.keras.models.load_model(model_path)
    st.success("Model loaded successfully!")
    

    num_images = st.slider("Number of images to generate", 1, 66, 16)
    noise_dim = model.input_shape[1]  # Assumes the generator takes noise as input
    
    if st.button("Generate Images"):
        noise = np.random.normal(0, 1, (num_images, noise_dim))
        generated_images = model.predict(noise)
        

        generated_images = (generated_images + 1) / 2  # Assuming tanh output
        

        fig, axes = plt.subplots(1, num_images, figsize=(num_images * 2, 2))
        for i, ax in enumerate(axes):
            ax.imshow(generated_images[i].squeeze(), cmap='gray')
            ax.axis("off")
        
        st.pyplot(fig)
except Exception as e:
    st.error(f"Error loading model: {e}")
