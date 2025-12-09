from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# 1. Load the trained model and the translators (encoders)
model = joblib.load('student_grade_model.pkl')
encoders = joblib.load('label_encoders.pkl')

# 2. Define the Home Page (Where users enter data)
@app.route('/')
def home():
    return render_template('index.html')

# 3. Define the Prediction Logic
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        
        # --- SAFE HANDLING FOR DEPARTMENT ---
        # The model only knows specific departments. If user picks "Other", 
        # we default to 'CSE' (or any common one) so the app doesn't crash.
        selected_dept = data['Department']
        known_departments = encoders['Department'].classes_
        
        if selected_dept not in known_departments:
            selected_dept = 'CSE' # Fallback value
        # ------------------------------------

        input_data = {
            'Gender': [data['Gender']],
            'Department': [selected_dept], # Use the safe variable here
            'SSC result': [data['SSC result']],
            'HSC result': [data['HSC result']],
            'Weekly Study Time at home': [data['Weekly Study Time at home']],
            'Attendance in class': [data['Attendance in class']]
        }
        
        df_input = pd.DataFrame(input_data)
        
        for col in df_input.columns:
            df_input[col] = encoders[col].transform(df_input[col])
            
        prediction_number = model.predict(df_input)
        
        target_col = 'Undergraduate 1st semester result (CGPA/GPA out of 4)'
        final_result = encoders[target_col].inverse_transform(prediction_number)
        
        return render_template('index.html', prediction_text=f'Predicted Result: {final_result[0]}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')
    
if __name__ == "__main__":
    app.run(debug=True)