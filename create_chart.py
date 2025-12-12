import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 1. Load your Dataset
df = pd.read_csv('Std_dataset.csv')

# 2. Convert Text to Numbers (Encoding)
# Correlation only works on numbers, so we turn "Male" into 1, etc.
df_encoded = df.copy()
label_encoders = {}

for col in df_encoded.columns:
    le = LabelEncoder()
    # We convert to string first to handle any mixed data types
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# 3. Calculate the Correlation Matrix
correlation_matrix = df_encoded.corr()

# 4. Filter for the "Top 15" most relevant features
# If we plot all 32 columns, the chart will be too messy to read.
target_col = 'Undergraduate 1st semester result (CGPA/GPA out of 4)'

# Get the top 15 columns that have the highest correlation with the Target
# (We use 'abs()' to find strong negative correlations too, like Smoking)
top_features = correlation_matrix[target_col].abs().sort_values(ascending=False).head(15).index

# Create a smaller matrix with only these top features
final_matrix = df_encoded[top_features].corr()

# 5. Draw the Chart
plt.figure(figsize=(12, 10))
sns.heatmap(final_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

# Add titles and labels
plt.title('Top 15 Factors Influencing Student Performance', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# 6. Save the Image
plt.tight_layout()
plt.savefig('correlation_heatmap.png')

print("✅ Success! Chart saved as 'correlation_heatmap.png'")