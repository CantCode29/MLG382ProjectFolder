import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import joblib  
import os
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


model_path = os.path.join(os.getcwd(), 'Artifacts', 'random_forest_model.pkl')
try:
    model = joblib.load(model_path)
    print(f"Model loaded successfully from: {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Student Grade Prediction (Random Forest)"),

   ##Input fields
    html.Div([
        html.Label("Age"),
        dcc.Input(id='age-input', type='number', value=16),
    ]),
    html.Div([
        html.Label("Gender (0: Male, 1: Female)"),
        dcc.Dropdown(id='gender-input', options=[{'label': 'Male', 'value': 0}, {'label': 'Female', 'value': 1}], value=0),
    ]),
    html.Div([
        html.Label("Ethnicity"),
        dcc.Dropdown(id='ethnicity-input', options=[{'label': 'Caucasian', 'value': 0}, {'label': 'African American', 'value': 1}, {'label': 'Asian', 'value': 2}, {'label': 'Other', 'value': 3}], value=0),
    ]),
    html.Div([
        html.Label("Parental Education"),
        dcc.Dropdown(id='parental-education-input', options=[{'label': 'None', 'value': 0}, {'label': 'High School', 'value': 1}, {'label': 'Some College', 'value': 2}, {'label': 'Bachelor\'s', 'value': 3}, {'label': 'Higher Study', 'value': 4}], value=0),
    ]),
    html.Div([
        html.Label("Study Time Weekly"),
        dcc.Input(id='study-time-input', type='number', value=10),
    ]),
    html.Div([
        html.Label("Absences"),
        dcc.Input(id='absences-input', type='number', value=5),
    ]),
    html.Div([
        html.Label("Tutoring (0: No, 1: Yes)"),
        dcc.Dropdown(id='tutoring-input', options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], value=0),
    ]),
    html.Div([
        html.Label("Parental Support"),
        dcc.Dropdown(id='parental-support-input', options=[{'label': 'None', 'value': 0}, {'label': 'Low', 'value': 1}, {'label': 'Moderate', 'value': 2}, {'label': 'High', 'value': 3}, {'label': 'Very High', 'value': 4}], value=0),
    ]),
    html.Div([
        html.Label("Extracurricular (0: No, 1: Yes)"),
        dcc.Dropdown(id='extracurricular-input', options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], value=0),
    ]),
    html.Div([
        html.Label("Sports (0: No, 1: Yes)"),
        dcc.Dropdown(id='sports-input', options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], value=0),
    ]),
    html.Div([
        html.Label("Music (0: No, 1: Yes)"),
        dcc.Dropdown(id='music-input', options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], value=0),
    ]),
    html.Div([
        html.Label("Volunteering (0: No, 1: Yes)"),
        dcc.Dropdown(id='volunteering-input', options=[{'label': 'No', 'value': 0}, {'label': 'Yes', 'value': 1}], value=0),
    ]),
    html.Div([
        html.Label("GPA"),
        dcc.Input(id='gpa-input', type='number', value=3.0, min=2.0, max=4.0),
    ]),

    html.Button('Predict Grade', id='predict-button', n_clicks=0),

    html.Div(id='prediction-output')
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
     State('volunteering-input', 'value'),
     State('gpa-input', 'value')]
)

def featureEng(inputData):
    df = inputData
    numeric_features =['Age',
    'StudyTimeWeekly',
    'Absences',
    ]
    ## scaling numeric features
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features]) 

    ## scaling categorical features
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    ordinal_encoder_Parent = OrdinalEncoder(categories=[[0,1,2,3,4]])
    ordinal_encoder_ParentalSup = OrdinalEncoder(categories=[[0,1,2,3,4]])

    ethnicity_encoded = ohe.fit_transform(df[['Ethnicity']])
    ethnicity_df = pd.DataFrame(ethnicity_encoded, columns=[f'Ethnicity_{cat}' for cat in ohe.categories_[0][1:]]) ## this is nominal so we split up each category into their own binary columns
    df = pd.concat([df, ethnicity_df], axis=1)
    df.drop('Ethnicity', axis=1, inplace=True)

    # Fit and transform 'ParentalEducation' using Ordinal Encoding
    df['ParentalEducation'] = ordinal_encoder_Parent.fit_transform(df[['ParentalEducation']]) ## these are ordinal so we dont split them up because their order matters 

    # Fit and transform 'ParentalSupport' using Ordinal Encoding
    df['ParentalSupport'] = ordinal_encoder_ParentalSup.fit_transform(df[['ParentalSupport']])
    return df

def predict_grade(n_clicks, age, gender, ethnicity, parental_education, study_time, absences, tutoring, parental_support, extracurricular, sports, music, volunteering, gpa):
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
        'GPA': [gpa]
    })

  
    prediction = model.predict(featureEng(input_data))[0]  # Get the single prediction

    
    grade_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'F'}
    predicted_grade = grade_mapping.get(prediction, 'Unknown')

    return f"Predicted Grade: {predicted_grade}"

if __name__ == '__main__':
    app.run(debug=True)