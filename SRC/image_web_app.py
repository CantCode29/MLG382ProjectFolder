import base64
import cv2
import numpy as np
import tensorflow as tf
from dash.dependencies import Input, Output
from dash import dcc, html
import dash
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=["assets/image_app.css"])
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


app.layout = html.Div([
    html.Header([
        html.Div([
            dcc.Markdown(
                '''
<svg width="50" height="30" viewBox="0 0 50 30">
    <path d="M1,10 L10,1 L19,10 L28,1 L37,10 L46,1" stroke="black" stroke-width="4" fill="none"></path>
</svg>
                ''',
                dangerously_allow_html=True
            )
        ], className="logo"),
        html.Nav([
            html.A("HOME", href="#"),
            html.A("SALE", href="#"),
            html.A("NEW & TRENDING", href="#")
        ], className="nav-main"),
        html.Div([
            html.Div([
                dcc.Input(placeholder="Search", type="text", style={
                    'border': 'none',
                    'backgroundColor': 'transparent',
                    'padding': '5px',
                    'width': '150px',
                    'outline': 'none'
                }),
                html.Div(
                    dcc.Markdown(
                        '''
<svg width="20" height="20" viewBox="0 0 24 24" fill="none">
    <circle cx="11" cy="11" r="8" stroke="currentColor" stroke-width="2" fill="none"></circle>
    <line x1="21" y1="21" x2="16.65" y2="16.65" stroke="currentColor" stroke-width="2"></line>
</svg>
                        ''',
                        dangerously_allow_html=True
                    ),
                    className="search-icon"
                )
            ], className="search-container"),
            html.Div([
                dcc.Markdown(
                    '''
<svg width="24" height="24" viewBox="0 0 24 24">
    <circle cx="12" cy="8" r="5" stroke="currentColor" stroke-width="2" fill="none"></circle>
    <path d="M20 21v-2a7 7 0 0 0-14 0v2" stroke="currentColor" stroke-width="2" fill="none"></path>
</svg>
                    ''',
                    dangerously_allow_html=True
                ),
                html.Span("1", className="notification")
            ], className="icon account"),
            html.Div([
                dcc.Markdown(
                    '''
<svg width="24" height="24" viewBox="0 0 24 24">
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78L12 21.23l8.84-8.84a5.5 5.5 0 0 0 0-7.78z"
          stroke="black" stroke-width="2" fill="none"></path>
</svg>
                    ''',
                    dangerously_allow_html=True
                )
            ], className="icon wishlist"),
            html.Div([
                dcc.Markdown(
                    '''
<svg width="24" height="24" viewBox="0 0 24 24">
    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"
          stroke="currentColor" stroke-width="2" fill="none"></rect>
</svg>
                    ''',
                    dangerously_allow_html=True
                )
            ], className="icon cart")
        ], className="nav-right")
    ]),
    html.Div([
        html.Div([
            html.Div("New Arrivals", className="tab active"),
            html.Div("Best Sellers", className="tab"),
            html.Div("New to Sale", className="tab")
        ], className="tab-buttons"),
        html.A("VIEW ALL", href="#", className="view-all")
    ], className="nav-tabs"),
    html.Div([
        # Product Card 1
        html.Div([
            html.Div([
                html.Img(src="/assets/img1.jpg",
                         alt="Originals Argyle Printed 1/4-Zip Sweatshirt"),
                html.Div([
                    dcc.Markdown(
                        '''
<svg width="20" height="20" viewBox="0 0 24 24">
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5
             0 0 0-7.78 7.78L12 21.23l8.84-8.84a5.5 5.5 0 0 0 0-7.78z"
          stroke="black" stroke-width="2" fill="none"></path>
</svg>
                        ''',
                        dangerously_allow_html=True
                    )
                ], className="favorite"),
                html.Div("$140", className="price")
            ], className="product-image"),
            html.Div([
                html.Div("Desigual Urban Chic Cream Backpack with Contrast Stitching",
                         className="product-title"),
                html.Div("Originals", className="product-category")
            ], className="product-info")
        ], className="product-card"),
        # Product Card 2
        html.Div([
            html.Div([
                html.Img(src="/assets/img2.jpg",
                         alt="Originals Argyle Printed 1/4-Zip Sweatshirt"),
                html.Div([
                    dcc.Markdown(
                        '''
<svg width="20" height="20" viewBox="0 0 24 24">
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5
             0 0 0-7.78 7.78L12 21.23l8.84-8.84a5.5 5.5 0 0 0 0-7.78z"
          stroke="black" stroke-width="2" fill="none"></path>
</svg>
                        ''',
                        dangerously_allow_html=True
                    )
                ], className="favorite"),
                html.Div("$140", className="price")
            ], className="product-image"),
            html.Div([
                html.Div("Luxury Braided Black Shoulder Bag with Gold Accent Chain",
                         className="product-title"),
                html.Div("Originals", className="product-category")
            ], className="product-info")
        ], className="product-card"),
        # Product Card 3
        html.Div([
            html.Div([
                html.Img(src="/assets/img3.jpg",
                         alt="Originals Argyle Printed 1/4-Zip Sweatshirt"),
                html.Div([
                    dcc.Markdown(
                        '''
<svg width="20" height="20" viewBox="0 0 24 24">
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5
             0 0 0-7.78 7.78L12 21.23l8.84-8.84a5.5 5.5 0 0 0 0-7.78z"
          stroke="black" stroke-width="2" fill="none"></path>
</svg>
                        ''',
                        dangerously_allow_html=True
                    )
                ], className="favorite"),
                html.Div("$140", className="price")
            ], className="product-image"),
            html.Div([
                html.Div("Elegant Brown Monogram Tote Bag with Leather Trim by BAGCO",
                         className="product-title"),
                html.Div("Originals", className="product-category")
            ], className="product-info")
        ], className="product-card"),
        # Product Card 4
        html.Div([
            html.Div([
                html.Img(src="/assets/img4.jpg",
                         alt="Originals Argyle Printed 1/4-Zip Sweatshirt"),
                html.Div([
                    dcc.Markdown(
                        '''
<svg width="20" height="20" viewBox="0 0 24 24">
    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5
             0 0 0-7.78 7.78L12 21.23l8.84-8.84a5.5 5.5 0 0 0 0-7.78z"
          stroke="black" stroke-width="2" fill="none"></path>
</svg>
                        ''',
                        dangerously_allow_html=True
                    )
                ], className="favorite"),
                html.Div("$140", className="price")
            ], className="product-image"),
            html.Div([
                html.Div("Chunky Knit Winter Beanie with Faux Fur Pom – Light Grey",
                         className="product-title"),
                html.Div("Originals", className="product-category")
            ], className="product-info")
        ], className="product-card"),
        html.Div([], className="divider")
    ], className="product-grid"),
    html.Div(
        style={
            'textAlign': 'center',
            'padding': '20px',
            'minHeight': '100vh',
            'display': 'flex',
            'flexDirection': 'column',
            'alignItems': 'center',
            'justifyContent': 'center',
        },
        children=[
            html.H1(
                "Become a Seller",
                style={'marginBottom': '20`px'}
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
        ])
])


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
