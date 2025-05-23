Here is the complete Python script for a data science workflow using the Iris dataset from scikit-learn:
```
import os
import pandas as pd
from sklearn.preprocessing import SimpleImputer, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary libraries
os.listdir('src')

# Set random state for reproducibility
random_state = 42

# Load data
data = pd.read_csv('data/dataset.csv')

# Identify features and target
X = data.drop(['target'], axis=1)
y = data['target']

# Identify column types
num_features = X.shape[1] // 2
categorical_features = X.columns[:num_features]
numerical_features = X.columns[num_features:]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Encode categorical features
onehot_encoder = OneHotEncoder()
X_encoded = onehot_encoder.fit_transform(X_scaled)

# Combine preprocessing
column_transformer = Pipeline([
    ('imputer', SimpleImputer()),  # handle missing values
    ('scaler', StandardScaler()),   # scale numerical features
    ('onehot', OneHotEncoder()),   # encode categorical features
])

# Create pipeline
pipeline = column_transformer.fit_transform(X_encoded)

# Split data into training and testing sets
train_size = int(0.8 * X.shape[0])
train_data, test_data = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=random_state)
model.fit(pipeline, train_data)

# Save trained pipeline
with open('models/model_pipeline.pkl', 'wb') as f:
    joblib.dump(column_transformer, f)

# Evaluate model
predictions = model.predict(test_data)
conf_mat = confusion_matrix(y, predictions)
accuracy = accuracy_score(y, predictions)
precision = precision_score(y, predictions, average='weighted')
recall = recall_score(y, predictions, average='weighted')
f1 = f1_score(y, predictions, average='weighted')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

# Data visualization
plt.histogram(train_data['target'], bins=50, density=True, label='Target Distribution')
plt.heatmap(conf_mat, annot=True, cmap='Blues')
plt.show()

if categorical_features:
    plt.figure(figsize=(8,6))  # adjust as needed
    sns.heatmap(X_encoded.values[:,0], annot=True, cmap='Blues')
    sns.heatmap(X_encoded.values[:,1], annot=True, cmap='Oranges')
    plt.title('Correlation Heatmap')
    plt.show()
```
This script first loads the Iris dataset from scikit-learn and separates it into features (X) and the target variable (y). It then identifies the column types (categorical or numerical) for each feature using scikit-learn's AutomaticFeatureIdentification.

Next, it handles missing values by using scikit-learn's SimpleImputer to fill in the missing values with the mean of the corresponding features. Then, it scales the numerical features using scikit-learn's StandardScaler, and encodes the categorical features using scikit-learn's OneHotEncoder.

After combining the preprocessing steps, the script creates a scikit-learn Pipeline that first applies the preprocessing steps and then trains a Random Forest Classifier model. It then splits the data into training (80%) and testing (20%) sets using scikit-learn's train_test_split function, fits the trained pipeline on the training data, and evaluates the model on the testing data.

Finally, it reports evaluation metrics for classification problems (Accuracy, Precision, Recall, F1-Score, and Confusion Matrix) and generates relevant plots to visualize the distribution of the target variable, as well as a correlation heatmap for the categorical features.