# %% [markdown]
# # Structured Data Classification Model (Sensor Activity Dataset)

# %% 
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# %% [markdown]
# ## Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--trainingdata", type=str, required=True, help='Path to the dataset')
args = parser.parse_args()

# %% [markdown]
# ## Load dataset
try:
    data = pd.read_csv(args.trainingdata, on_bad_lines='skip')
except Exception as e:
    print(f"Error reading the dataset: {e}")
    exit(1)

print("Dataset preview:")
print(data.head())
print("\nColumns in the dataset:")
print(data.columns)

if 'Activity' not in data.columns:
    print("The dataset must contain an 'Activity' column as the target variable.")
    exit(1)

# %% [markdown]
# ## Preprocess dataset
X = data.drop(columns=['Activity', 'subject'], axis=1)
y = data['Activity']

if X.isnull().any().any():
    print("Warning: Dataset contains missing values. Dropping missing rows.")
    data = data.dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# ## Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# %% [markdown]
# ## Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

# Fallback visualization using matplotlib
print("Confusion Matrix:")
print(cm)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
