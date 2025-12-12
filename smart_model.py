import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load Data
df = pd.read_csv('Std_dataset.csv')
target = 'Undergraduate 1st semester result (CGPA/GPA out of 4)'

# 2. Define the Top 5 Scientifically Proven Features
features = [
    'HSC result', 
    'Smoking', 
    'Taking Exam Preparation', 
    'Attend any Seminar related to department', 
    'Scholarship in SSC'
]

# 3. Simplify Target (High/Avg/Low) - Same as before for consistency
def simplify_grade(grade):
    if grade in ['3.75 - 4.00', '3.50 - 3.74', 'GPA 5.00']:
        return 'High Performance (3.50+)'
    elif grade in ['3.25 - 3.49', '3.00 - 3.24']:
        return 'Average Performance (3.00 - 3.49)'
    else:
        return 'Needs Improvement (< 3.00)'

df['Simple_Target'] = df[target].apply(simplify_grade)

# 4. Encode Features
encoders = {}
for col in features:
    le = LabelEncoder()
    # Force to string to handle any mixed types
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Encode Target
target_le = LabelEncoder()
df['Target_Encoded'] = target_le.fit_transform(df['Simple_Target'])
encoders['Target_Final'] = target_le 

# 5. Train
X = df[features]
y = df['Target_Encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 6. Save as "smart_model.pkl" so we don't overwrite the old one
joblib.dump(model, './models/smart_model.pkl')
joblib.dump(encoders, './models/smart_encoders.pkl')

print("Success! 'smart_model.pkl' created.")