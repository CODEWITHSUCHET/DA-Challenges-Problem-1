# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from sklearn.preprocessing import StandardScaler

# Loading the dataset (replace this with your dataset)
# df = pd.read_csv('your_dataset.csv')

# Example dataset
# Assuming columns: 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'target'
df = pd.DataFrame({
    'feature_1': np.random.randn(100),  # Example numeric feature
    'feature_2': np.random.choice(['A', 'B', 'C'], 100),  # Example categorical feature
    'feature_3': np.random.choice([0, 1], 100),  # Example binary feature
    'feature_4': np.random.rand(100),  # Example numeric feature
    'target': np.random.choice([0, 1], 100)  # Binary target variable
})

# Data Preparation
# Encoding categorical features if necessary
df = pd.get_dummies(df, drop_first=True)

# Splitting the data into features (X) and target (y)
X = df.drop(columns='target')
y = df['target']

# Splitting the dataset into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (optional, but generally good for models like SVM, Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a list of models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(),
    'Gradient Boosting': GradientBoostingClassifier()
}

# Store the results of model performance
results = []

# Model Training and Evaluation
for model_name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) if len(np.unique(y)) > 2 else None  # R² score for regression-like tasks
    
    # Append the results
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'R² Score': r2
    })

# Create a DataFrame to hold the results
results_df = pd.DataFrame(results)

# Hyperparameter Tuning using GridSearchCV
tuned_results = []
param_grid = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Decision Tree': {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]},
    'Random Forest': {'n_estimators': [50, 100, 200]},
    'Support Vector Machine': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.5]}
}

for model_name, model in models.items():
    grid = GridSearchCV(model, param_grid[model_name], cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    
    # Get the best model and evaluate on the test set
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) if len(np.unique(y)) > 2 else None  # R² score for regression-like tasks
    
    tuned_results.append({
        'Model': model_name + ' (Tuned)',
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'R² Score': r2
    })

# Create a DataFrame to hold the tuned results
tuned_results_df = pd.DataFrame(tuned_results)

# Combine results for comparison
final_results = pd.concat([results_df, tuned_results_df], ignore_index=True)

# Display the comparison table
print(final_results)

# Visualization: Comparing performance metrics
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R² Score']
final_results = final_results.set_index('Model')

plt.figure(figsize=(12, 8))
final_results[metrics].plot(kind='bar', figsize=(12, 8), width=0.8)
plt.title('Comparison of Model Performance')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
