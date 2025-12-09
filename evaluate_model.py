import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Data
data = pd.read_csv('Std_dataset.csv')

# 2. Select Features
features = ['Gender', 'Department', 'SSC result', 'HSC result', 'Weekly Study Time at home', 'Attendance in class']
target = 'Undergraduate 1st semester result (CGPA/GPA out of 4)'

df = data[features + [target]].copy()

# 3. Preprocessing (Convert Text to Numbers)
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    encoders[column] = le

X = df[features]
y = df[target]

# 4. Split Data (80% for Training, 20% for Testing)
# This is where we hide some data to test the student (AI) later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Test the Model
# We ask the model to predict the answers for the 20% of data it has never seen
y_pred = model.predict(X_test)

# 7. Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"------------------------------------------------")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"------------------------------------------------")

# 8. Detailed Report
# We convert the numbers back to text labels (e.g., 0 -> 'Fail') for the report
target_names = encoders[target].classes_
print("\nClassification Report (Detailed Performance):")
print(classification_report(y_test, y_pred, target_names=target_names))

# Optional: Plot the Confusion Matrix (Who got confused with what?)
# This saves an image you can put in your project report
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names, cmap='Blues')
plt.ylabel('Actual Result')
plt.xlabel('Predicted Result')
plt.title('Confusion Matrix: Where did the AI make mistakes?')
plt.savefig('accuracy_chart.png')
print("\nChart saved as 'accuracy_chart.png'")