import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import tensorflow as tf
import numpy as np
import cv2
import base64
import os

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Load the pre-trained model
ARTIFACTS_DIR = os.path.join(os.getcwd(), 'Artifacts')
MODEL_DIR = os.path.join(ARTIFACTS_DIR, 'imageprediction.h5')
IMG_SIZE = (80, 60)
try:
    model = tf.keras.models.load_model(MODEL_DIR)
    print(f"Model loaded successfully from: {MODEL_DIR}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


def load_and_process_images(image_array):  # DONT DELETE THIS FUNCTION
    try:
        print(f"--- Inside load_and_process_image_for_prediction ---")

        print("Reading cv2 image")
        img = cv2.resize(image_array, IMG_SIZE,
                         interpolation=cv2.INTER_LANCZOS4)
        print(f"After cv2.resize: Shape: {img.shape}, dtype: {img.dtype}")
        if img.dtype == np.float64:
            # Scale to 0 - 255 and convert to uint8
            print(
                f"Image dtype is {img.dtype}, attempting conversion to uint8.")
            img_converted = (img * 255).astype(np.uint8)
            return img_converted / 255

        elif img.dtype == np.uint8:
            return img

    except:
        print("Im returning None you bozo")
        return None, None


app.layout = html.Div(
    style={
        'textAlign': 'center',
        'padding': '20px',
        'backgroundColor': '#f0f0f0',
        'minHeight': '100vh',
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'justifyContent': 'center',
    },
    children=[
        html.H1(
            "Women's Accessory Image Classifier",
            style={'color': '#333', 'marginBottom': '10px'}
        ),
        html.P(
            "Upload an image to predict if it's a women's accessory (e.g., jewelry, handbags) or not.",
            style={'color': '#555', 'marginBottom': '20px', 'fontSize': '18px'}
        ),

    \
        dcc.Upload(
            id='upload-image',
            children=html.Button(
                'Upload Image',
                style={
                    'backgroundColor': '#4CAF50',
                    'color': 'white',
                    'fontSize': '20px',
                    'padding': '15px 30px',
                    'border': 'none',
                    'borderRadius': '5px',
                    'cursor': 'pointer',
                    'boxShadow': '0 2px 5px rgba(0,0,0,0.2)',
                    'transition': 'background-color 0.3s',
                },
            ),
            multiple=False,
            style={'marginBottom': '30px'}
        ),

        # Div to display the uploaded image and prediction
        html.Div(id='output-image-upload', style={'marginTop': '20px'}),
    ]
)


@app.callback(  # dont delete this either
    Output('output-image-upload', 'children'),
    Input('upload-image', 'contents')
)
def update_output(uploaded_image):  # DONT DELETE THIS FUNCTION
    if uploaded_image is None:
        return html.P("No image uploaded yet.", style={'color': '#888'})

    if model is None:
        return html.P("Model not loaded. Cannot make predictions.", style={'color': 'red'})

    try:
        content_type, content_string = uploaded_image.split(',')
        if not content_string:
            return html.P("Error: Image data is empty. Please upload a valid image file.", style={'color': 'red'})

        decoded = base64.b64decode(content_string)

        # Convert bytes to numpy array and decode image
        image_array = np.frombuffer(decoded, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        # Check if image decoding was successful
        if image is None:
            return html.P("Error: Could not decode the image. Please upload a valid image file (e.g., PNG, JPEG).", style={'color': 'red'})

        # Process the image for prediction
        processed_image = load_and_process_images(image)

        processed_image = np.expand_dims(processed_image, axis=0)

        # Make prediction
        prediction_prob = model.predict(processed_image)[0][0]
        prediction = "Women's Accessory" if prediction_prob > 0.5 else "Not a Women's Accessory"
        confidence = prediction_prob if prediction_prob > 0.5 else 1 - prediction_prob
        confidence = confidence * 100
        image_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.png', cv2.cvtColor(
            image_display, cv2.COLOR_RGB2BGR))  # Encode as BGR for png
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        image_data = f"data:image/png;base64,{image_base64}"

        # Return the image and prediction
        return html.Div([
            html.Img(src=image_data, style={
                     'width': '300px', 'height': 'auto', 'marginBottom': '20px'}),
            html.H4(f"Prediction: {prediction}", style={'color': '#333'}),
            html.P(f"Confidence: {confidence:.2f}%", style={
                   'color': '#555', 'fontSize': '16px'})
        ])

    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()
        return html.P(f"An unexpected error occurred: {str(e)}. Please check the console for details.", style={'color': 'red'})


if __name__ == '__main__':
    app.run(debug=True)
