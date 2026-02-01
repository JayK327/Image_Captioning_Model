# streamlit_app.py
import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.layers import add
from tensorflow.keras.models import Model

# -------------------------------
# Utility functions
# -------------------------------
def extract_features(image, model):
    try:
        image = image.resize((299, 299))
        image = np.array(image)
        if image.shape[2] == 4:  # Convert 4-channel images to 3 channels RGBA->RGB
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image / 127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature
    except:
        st.error("ERROR: Couldn't process the image! Make sure it's a valid image file.")
        return None

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo, sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text.replace('start', '').replace('end', '').strip()

def define_model(vocab_size, max_length):
    # Image feature input
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence input
    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Flickr8k Image Caption Generator")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    st.text("Generating caption... Please wait!")

    # Load tokenizer and models (ensure these files exist in the same directory)
    max_length = 32
    tokenizer = load(open("tokenizer.p", "rb"))
    vocab_size = len(tokenizer.word_index) + 1

    # Load trained caption model
    model = define_model(vocab_size, max_length)
    model.load_weights("models/model_9.keras")

    # Load Xception model for feature extraction
    xception_model = Xception(include_top=False, pooling="avg")

    # Extract features and generate caption
    photo = extract_features(img, xception_model)
    if photo is not None:
        description = generate_desc(model, tokenizer, photo, max_length)
        st.success("Caption generated successfully!")
        st.write(f"**Caption:** {description}")
