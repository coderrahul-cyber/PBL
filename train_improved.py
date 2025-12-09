import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the Data
data = pd.read_csv('Std_dataset.csv')
features = ['Gender', 'Department', 'SSC result', 'HSC result', 'Weekly Study Time at home', 'Attendance in class']
target = 'Undergraduate 1st semester result (CGPA/GPA out of 4)'

# 2. THE FIX: Simplify the Target Variable
# We group the specific grades into 3 distinct "Performance Levels"
def simplify_grade(grade):
    # High Grades (3.50 and above)
    if grade in ['3.75 - 4.00', '3.50 - 3.74', 'GPA 5.00']:
        return 'High Performance (3.50+)'
    # Average Grades (3.00 to 3.49)
    elif grade in ['3.25 - 3.49', '3.00 - 3.24']:
        return 'Average Performance (3.00 - 3.49)'
    # Low Grades (Below 3.00)
    else:
        return 'Needs Improvement (< 3.00)'

# Apply the simplification
df = data[features + [target]].copy()
df['Simple_Target'] = df[target].apply(simplify_grade)

# 3. Preprocessing (Convert Text to Numbers)
encoders = {}
for col in features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Encode the NEW target
target_le = LabelEncoder()
df['Simple_Target_Encoded'] = target_le.fit_transform(df['Simple_Target'])
encoders[target] = target_le # Save this so the App knows the new labels

# 4. Train/Test Split
X = df[features]
y = df['Simple_Target_Encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model
# We use 'class_weight="balanced"' to handle the fact that some groups have fewer students
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 6. Check New Accuracy
y_pred = model.predict(X_test)
new_accuracy = accuracy_score(y_test, y_pred)
print("------------------------------------------------")
print(f"Old Accuracy: ~29.71%")
print(f"New Model Accuracy: {new_accuracy * 100:.2f}%")
print("------------------------------------------------")
print("\nNew Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_le.classes_))

# 7. Save the new "Brain"
joblib.dump(model, 'student_grade_model.pkl')
joblib.dump(encoders, 'label_encoders.pkl')
print("\nSuccess! New model saved. Restart your app to see the changes.")