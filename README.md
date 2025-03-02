Churn Prediction Analysis - Step-by-Step Guide
# 1. Introduction
## Objective:
The goal of this project is to analyze customer churn data, identify key factors influencing churn, and develop a
predictive model.
## Dataset Overview:
- Customer Demographics: State, Account Length, Area Code
- Service Plans: International Plan, Voice Mail Plan
- Usage Statistics: Total Day, Evening, Night, and International Minutes, Calls, and Charges
- Customer Service Interactions: Customer Service Calls
- Target Variable: Churn
# 2. Data Preprocessing
## Step 1: Load the Dataset
```python
import pandas as pd
data = pd.read_csv('churn-bigml-80.csv')
print(data.head())
```
## Step 2: Handle Missing Values
```python
print(data.isnull().sum())
data = data.dropna()
```
## Step 3: Remove Duplicates
```python
data = data.drop_duplicates()
```
# 3. Exploratory Data Analysis (EDA)
## Step 5: Summary Statistics
```python
print(data.describe())
```
## Step 6: Visualizations
```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(data['Total day minutes'], bins=30, kde=True)
plt.title('Distribution of Total Day Minutes')
plt.show()
```
## Step 7: Correlation Analysis
```python
numeric_data = data.select_dtypes(include=['number'])
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5, cbar=True)
plt.title('Feature Correlations')
plt.show()
```
# 4. Feature Engineering
## Step 8: Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
features = ['Account length', 'Total day minutes', 'Total day calls', 'Total day charge',
'Total eve minutes', 'Total eve calls', 'Total eve charge', 'Total night minutes',
'Total night calls', 'Total night charge', 'Total intl minutes', 'Total intl calls',
'Total intl charge', 'Customer service calls']
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])
```
# 5. Model Training & Evaluation
## Step 9: Train-Test Split
```python
from sklearn.model_selection import train_test_split
X = data[features]
y = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Step 10: Train Model
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
## Step 11: Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```
# 6. Power BI Dashboard Recommendations
- Churn Rate Analysis (Pie Chart)
- Total Charges vs. Churn Status (Boxplot)
- Customer Service Calls vs. Churn (Bar Chart)
- Feature Importance from Model (Bar Chart)
