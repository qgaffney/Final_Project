import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Import the Dataset
# Replace 'your_dataset.csv' with the actual file name or path
df = pd.read_csv('student_habits_performance.csv')

# Preview the dataset
print("Dataset Preview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# Step 2.1: Preprocessing
# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values (fill with mean for numeric columns)
for col in ['study_hours_per_day', 'social_media_hours', 'netflix_hours', 'sleep_hours']:
    df[col].fillna(df[col].mean(), inplace=True)

# Ensure exam_score has no missing values
df['exam_score'].fillna(df['exam_score'].mean(), inplace=True)

# Step 3: Select Feature and Target Variables
# Replace 'feature_columns' and 'target_column' with actual column names from your dataset
features = df[['study_hours_per_day', 'social_media_hours', 'netflix_hours', 'sleep_hours']]
target = df['exam_score']

# Scale features using StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)

# Step 4: Split the Dataset (70% Training, 30% Testing)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

# Step 5: Initialize and Fit a Decision Tree Model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 6: Make Predictions on the Test Set
y_pred = model.predict(X_test)

# Step 7: Compute and Print Regression Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nRegression Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Step 8: Visualize Predicted vs. Actual Exam Scores
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Exam Score')
plt.ylabel('Predicted Exam Score')
plt.title('Predicted vs. Actual Exam Scores')
plt.tight_layout()
plt.show()

# Step 9: Apply K-Means Clustering
# Apply K-Means (2 clusters for simplicity; adjust based on data)
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(features_scaled)

# Step 10: Visualize Clustering Results
# Scatter plot of two features with cluster assignments and exam_score as color
plt.figure(figsize=(8, 6))
scatter = sns.scatterplot(x=features_scaled_df['study_hours_per_day'], 
                         y=features_scaled_df['sleep_hours'],
                         hue=df['cluster'], 
                         palette='Set1', 
                         size=df['exam_score'], 
                         sizes=(20, 200))
plt.title('K-Means Clustering of Student Habits')
plt.xlabel('Scaled Study Hours per Day')
plt.ylabel('Scaled Sleep Hours')
plt.legend(title='Cluster & Exam Score')
plt.tight_layout()
plt.show()

# Step 11: Interpret Clustering Results
print("\nCluster Centers (Original Scale):")
print(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features.columns))
print("\nCluster Counts:")
print(df['cluster'].value_counts())
print("\nAverage Exam Score per Cluster:")
print(df.groupby('cluster')['exam_score'].mean())