import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load your specific dataset
data = pd.read_csv('Std_dataset.csv')

# 2. Select the most relevant features (Inputs)
# We pick 6 clear factors to keep the project simple but effective.
# You can add more if you want, but you'll need to add them to the website later too.
features = [
    'Gender', 
    'Department', 
    'SSC result', 
    'HSC result', 
    'Weekly Study Time at home', 
    'Attendance in class'
]
target = 'Undergraduate 1st semester result (CGPA/GPA out of 4)'

# Filter the data to keep only what we need
df = data[features + [target]].copy()

# 3. Preprocessing: Convert Text to Numbers
# Machine Learning only understands numbers, so we use "LabelEncoder"
# to turn "Male" -> 1, "Female" -> 0, "GPA 5.00" -> 5, etc.

encoders = {} # We save these to translate user input later

for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    encoders[column] = le # Save the translator for this column

# 4. Split into Training (Input) and Testing (Target)
X = df[features]
y = df[target]

# 5. Train the Model
# We use RandomForestClassifier because it's great for categorical data (text-based choices)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 6. Save the Model and the Encoders
# We need the encoders to translate the user's input on the website back to numbers
joblib.dump(model, 'student_grade_model.pkl')
joblib.dump(encoders, 'label_encoders.pkl')

print("Success! Model trained.")
print("Files created: 'student_grade_model.pkl' and 'label_encoders.pkl'")