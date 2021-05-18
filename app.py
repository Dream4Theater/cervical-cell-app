import SessionState
import streamlit as st
import tensorflow as tf

model = tf.keras.models.load_model('MobileNet.h5')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

class_names = [
    'Dyskeratotic',
    'Koilocytotic',
    'Metaplastic',
    'Parabasal',
    'Superficial-Intermediate']

### Streamlit code (works as a straigtht-forward script) ###
st.title("Diagnostic Classification of Cervical Cell Images")
@st.cache # cache the function so predictions aren't always redone (Streamlit refreshes every click)
def make_prediction(image):

    img = tf.image.decode_bmp(image, channels=3)
    img = tf.image.resize(img,[224,224])
    img /= 255.
    img = tf.cast(tf.expand_dims(img, axis=0), tf.float32)

    preds = model.predict(img)
    pred_class = class_names[tf.argmax(preds)]
    pred_conf = tf.reduce_max(preds)
    return image, pred_class, pred_conf

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Upload an image",
                                 type=["bmp"])

#buttuon to trigger prediction
session_state = SessionState.get(pred_button=False)

# Create logic for app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True 

if session_state.pred_button:
    session_state.image, session_state.pred_class, session_state.pred_conf = make_prediction(session_state.uploaded_image)
    st.write(f"Prediction: {session_state.pred_class[0]}, \
               Confidence: {session_state.pred_conf[0]:.2f}")