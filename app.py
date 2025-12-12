from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# --- LOAD MODEL 1: The Manual One (Old) ---
try:
    model_manual = joblib.load('./modelsstudent_grade_model.pkl')
    encoders_manual = joblib.load('./models/label_encoders.pkl')
except:
    print("Warning: Manual model not found. Run train_improved.py first.")

# --- LOAD MODEL 2: The Smart One (New) ---
try:
    model_smart = joblib.load('./models/smart_model.pkl')
    encoders_smart = joblib.load('./models/smart_encoders.pkl')
except:
    print("Warning: Smart model not found. Run train_smart.py first.")


# ================= ROUTE 1: MANUAL MODEL =================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        
        # Safe Department Handling
        selected_dept = data['Department']
        if selected_dept not in encoders_manual['Department'].classes_:
            selected_dept = 'CSE' 

        input_data = {
            'Gender': [data['Gender']],
            'Department': [selected_dept],
            'SSC result': [data['SSC result']],
            'HSC result': [data['HSC result']],
            'Weekly Study Time at home': [data['Weekly Study Time at home']],
            'Attendance in class': [data['Attendance in class']]
        }
        
        df = pd.DataFrame(input_data)
        for col in df.columns:
            df[col] = encoders_manual[col].transform(df[col])
            
        pred = model_manual.predict(df)
        
        # Use a generic decoder if the specific target key varies
        # We assume the target encoder is stored under the target name or a specific key
        target_name = 'Undergraduate 1st semester result (CGPA/GPA out of 4)'
        # If we used train_improved.py, the key might be the column name itself
        res = encoders_manual[target_name].inverse_transform(pred)
        
        return render_template('index.html', prediction_text=f'Predicted Result: {res[0]}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


# ================= ROUTE 2: SMART MODEL =================
@app.route('/smart')
def smart_home():
    return render_template('smart.html')

@app.route('/predict_smart', methods=['POST'])
def predict_smart():
    try:
        data = request.form
        
        # The 5 Smart Features
        input_data = {
            'HSC result': [data['HSC result']],
            'Smoking': [data['Smoking']],
            'Taking Exam Preparation': [data['Taking Exam Preparation']],
            'Attend any Seminar related to department': [data['Attend any Seminar related to department']],
            'Scholarship in SSC': [data['Scholarship in SSC']]
        }
        
        df = pd.DataFrame(input_data)
        
        # Encode using the SMART encoders
        for col in df.columns:
            df[col] = encoders_smart[col].transform(df[col])
            
        pred = model_smart.predict(df)
        
        # Decode using the stored 'Target_Final' encoder
        res = encoders_smart['Target_Final'].inverse_transform(pred)
        
        return render_template('smart.html', prediction_text=f'Smart Prediction: {res[0]}')

    except Exception as e:
        return render_template('smart.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)