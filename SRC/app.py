import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder


ARTIFACTS_DIR = os.path.join(os.getcwd(), 'Artifacts')
MODEL_DIR = os.path.join(ARTIFACTS_DIR, 'Logisticregression.pkl')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
OHE_ETHNICITY_PATH = os.path.join(ARTIFACTS_DIR, 'ohe_ethnicity.pkl')
ORDINAL_PARENT_EDU_PATH = os.path.join(ARTIFACTS_DIR, 'ordinal_parent_edu.pkl')
ORDINAL_PARENT_SUP_PATH = os.path.join(ARTIFACTS_DIR, 'ordinal_parent_sup.pkl')

def load_artifact(path, description):
    """Loads a joblib artifact from a given path."""
    try:
        artifact = joblib.load(path)
        print(f"{description} loaded successfully from: {path}")
        return artifact
    except FileNotFoundError:
        print(f"Error loading {description}: File not found at {path}. Please ensure the artifact exists.")
        return None
    except Exception as e:
        print(f"Error loading {description} from {path}: {e}")
        return None


model = load_artifact(MODEL_DIR, "Model")
scaler = load_artifact(SCALER_PATH, "Scaler")
ohe_ethnicity = load_artifact(OHE_ETHNICITY_PATH, "Ethnicity OneHotEncoder")
ordinal_encoder_parent_edu = load_artifact(ORDINAL_PARENT_EDU_PATH, "Parental Education OrdinalEncoder")
ordinal_encoder_parent_sup = load_artifact(ORDINAL_PARENT_SUP_PATH, "Parental Support OrdinalEncoder")

def featureEng(input_df):
    """
    Applies pre-fitted transformations to the input DataFrame.

    Args:
        input_df (pd.DataFrame): DataFrame with raw input features.

    Returns:
        pd.DataFrame: DataFrame with transformed features, ready for prediction.
                      Returns None if any required artifact is missing.
    """
    
    EXPECTED_MODEL_COLUMNS = [
        'Age', 'Gender', 'ParentalEducation', 'StudyTimeWeekly', 'Absences',
        'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music',
        'Volunteering', 'Ethnicity_1', 'Ethnicity_2', 'Ethnicity_3'
    ]

    
    if not all([scaler, ohe_ethnicity, ordinal_encoder_parent_edu, ordinal_encoder_parent_sup]):
        print("Error: One or more preprocessing artifacts are not loaded. Cannot perform feature engineering.")
        return None

    df = input_df.copy() 

    
    numeric_features_to_scale = ['Age', 'StudyTimeWeekly', 'Absences']
    nominal_features = ['Ethnicity']
    ordinal_features_parent_edu = ['ParentalEducation']
    ordinal_features_parent_sup = ['ParentalSupport']

    
    try:
        
        if numeric_features_to_scale:
            df[numeric_features_to_scale] = scaler.transform(df[numeric_features_to_scale])
            print("Applied StandardScaler transform.")

        
        if ordinal_features_parent_edu and 'ParentalEducation' in df.columns:
             df['ParentalEducation'] = ordinal_encoder_parent_edu.transform(df[ordinal_features_parent_edu])
             print("Applied Parental Education OrdinalEncoder transform.")
        elif ordinal_features_parent_edu:
             print("Warning: 'ParentalEducation' column not found in input data for ordinal encoding.")

        
        if ordinal_features_parent_sup and 'ParentalSupport' in df.columns:
             df['ParentalSupport'] = ordinal_encoder_parent_sup.transform(df[ordinal_features_parent_sup])
             print("Applied Parental Support OrdinalEncoder transform.")
        elif ordinal_features_parent_sup:
             print("Warning: 'ParentalSupport' column not found in input data for ordinal encoding.")

        
        if nominal_features and 'Ethnicity' in df.columns:
            ethnicity_encoded = ohe_ethnicity.transform(df[nominal_features])
            ohe_feature_names = ohe_ethnicity.get_feature_names_out(nominal_features)
            ethnicity_df = pd.DataFrame(ethnicity_encoded, columns=ohe_feature_names, index=df.index)
            df = pd.concat([df.drop(nominal_features, axis=1), ethnicity_df], axis=1)
            print("Applied Ethnicity OneHotEncoder transform.")
        elif nominal_features:
            print("Warning: 'Ethnicity' column not found in input data for one-hot encoding.")


        
        for col in ['Ethnicity_1', 'Ethnicity_2', 'Ethnicity_3']:
            if col not in df.columns:
                df[col] = 0

       
        missing_cols = set(EXPECTED_MODEL_COLUMNS) - set(df.columns)
        if missing_cols:
            print(f"Error: Missing columns after feature engineering: {missing_cols}")
            return None

        final_df = df[EXPECTED_MODEL_COLUMNS]
        print(f"Final columns for prediction: {final_df.columns.tolist()}")
        return final_df

    except KeyError as e:
        print(f"Error during transformation: Missing column {e}. Ensure input data has all required columns.")
        return None
    except ValueError as e:
        print(f"Error during transformation: {e}. Check if input values are valid for the encoders/scaler.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during feature engineering: {e}")
        return None


app = dash.Dash(__name__, title="Bright Academy Student Grade Prediction")
server = app.server


app.layout = html.Div(className='container', children=[
    html.H1("Bright Academy Student Grade Prediction", className='title'),

    
    html.Div(className='explanation-text', children=[
        html.P(
            """
            This dashboard utilizes a predictive model to estimate a student's Grade Class at Bright Academy.
            The Grade Class represents the student's academic performance level (A, B, C, D, or F).
            Please note that Grade Point Average (GPA) is excluded from the prediction features as it is the numerical basis from which the Grade Class is derived.
            """
        ),
        html.P("The model's prediction is based on a range of student characteristics and behaviors, including:"),
        html.Ul([
            html.Li("Demographic details: Age, Gender (Male/Female), and Ethnicity (Caucasian, African American, Asian, Other)."),
            html.Li("Parental Information: Parental Education level (None to Higher Study) and the level of Parental Support (None to Very High)."),
            html.Li("Study Habits: Weekly Study Time, number of Absences during the school year, and participation in Tutoring (Yes/No)."),
            html.Li("Extracurricular Activities: Participation in Extracurricular activities overall (Yes/No), specifically including Sports (Yes/No), Music (Yes/No), and Volunteering (Yes/No).")
        ]),
         html.P("By inputting these details, users can receive a predicted Grade Class score.")
    ]), 

    html.Div(className='content-wrapper', children=[ #
        
        html.Div(className='left-column', children=[
            html.H2("Highest Predictor for Grade Class: Absences", className='graph-title'), 

            
            html.Img(
                src='/assets/absences_gpa_graph.png', 
                alt='Line graph showing the relationship between absences and GPA', 
                className='graph-image' 
            ),

            
            html.Div(className='summary-text', children=[
                html.P(
                    "Based on our modeling analysis using several different models, absences from class were consistently identified as the most significant predictor of student GPA and overall grade class scores. This highlights the critical importance of regular attendance for academic success at Bright Academy."
                ),
                
                html.A(
                    "For more information, please click here to download the full report.", 
                    href='/assets/bright_academy_report.pdf', 
                    download='bright_academy_report.pdf', 
                    className='download-link' 
                )
            ])
        ]), 

        
        html.Div(className='right-column', children=[
             html.H2("Enter Student Information", className='input-form-title'), 
            html.Div(className='input-section', children=[
                html.Div(className='input-group', children=[
                    html.Label("Age", className='label'),
                    dcc.Input(id='age-input', type='number', value=16, className='input-field'),
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Gender", className='label'),
                    dcc.Dropdown(id='gender-input', options=[{'label': 'Male', 'value': 0}, {'label': 'Female', 'value': 1}], value=0, className='dropdown'),
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Ethnicity", className='label'),
                    dcc.Dropdown(id='ethnicity-input', options=[{'label': 'Caucasian', 'value': 0}, {'label': 'African American', 'value': 1}, {'label': 'Asian', 'value': 2}, {'label': 'Other', 'value': 3}], value=0, className='dropdown'),
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Parental Education", className='label'),
                    dcc.Dropdown(id='parental-education-input', options=[{'label': 'None', 'value': 0}, {'label': 'High School', 'value': 1}, {'label': 'Some College', 'value': 2}, {'label': 'Bachelor\'s', 'value': 3}, {'label': 'Higher Study', 'value': 4}], value=0, className='dropdown'),
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Study Time Weekly", className='label'),
                    dcc.Input(id='study-time-input', type='number', value=10, className='input-field'),
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Absences", className='label'),
                    dcc.Input(id='absences-input', type='number', value=5, className='input-field'),
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Tutoring", className='label'),
                    dcc.Dropdown(id='tutoring-input', options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], value=0, className='dropdown'),
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Parental Support", className='label'),
                    dcc.Dropdown(id='parental-support-input', options=[{'label': 'None', 'value': 0}, {'label': 'Low', 'value': 1}, {'label': 'Moderate', 'value': 2}, {'label': 'High', 'value': 3}, {'label': 'Very High', 'value': 4}], value=0, className='dropdown'),
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Extracurricular", className='label'),
                    dcc.Dropdown(id='extracurricular-input', options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], value=0, className='dropdown'),
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Sports", className='label'),
                    dcc.Dropdown(id='sports-input', options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], value=0, className='dropdown'),
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Music", className='label'),
                    dcc.Dropdown(id='music-input', options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], value=0, className='dropdown'),
                ]),
                html.Div(className='input-group', children=[
                    html.Label("Volunteering", className='label'),
                    dcc.Dropdown(id='volunteering-input', options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], value=0, className='dropdown'),
                ])
            ]), 

            html.Button('Predict Grade', id='predict-button', n_clicks=0, className='predict-button'),

            html.Div(id='prediction-output', className='output-section')
        ])
    ]) 
])

@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('age-input', 'value'),
     State('gender-input', 'value'),
     State('ethnicity-input', 'value'),
     State('parental-education-input', 'value'),
     State('study-time-input', 'value'),
     State('absences-input', 'value'),
     State('tutoring-input', 'value'),
     State('parental-support-input', 'value'),
     State('extracurricular-input', 'value'),
     State('sports-input', 'value'),
     State('music-input', 'value'),
     State('volunteering-input', 'value')]
)
def predict_grade(n_clicks, age, gender, ethnicity, parental_education, study_time, absences, tutoring, parental_support, extracurricular, sports, music, volunteering):
    """Callback to trigger grade prediction based on input values."""
    if n_clicks == 0:
        return ""

    
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Ethnicity': [ethnicity],
        'ParentalEducation': [parental_education],
        'StudyTimeWeekly': [study_time],
        'Absences': [absences],
        'Tutoring': [tutoring],
        'ParentalSupport': [parental_support],
        'Extracurricular': [extracurricular],
        'Sports': [sports],
        'Music': [music],
        'Volunteering': [volunteering],
        'GPA': 0 
    })


    prediction_df = featureEng(input_data)

    if prediction_df is None:
        return "Error: Could not prepare data for prediction. Check server logs."

    
    if model is None:
         return "Error: Model not loaded. Cannot make prediction."

    try:
        prediction = model.predict(prediction_df)[0]  
        
        grade_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
        predicted_grade = grade_mapping.get(prediction, 'Unknown')

        return f"Predicted Grade: {predicted_grade}"
    except Exception as e:
        return f"Error during prediction: {e}"


if __name__ == '__main__':
    
    app.run(debug=True)

