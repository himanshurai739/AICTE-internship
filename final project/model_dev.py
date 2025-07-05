import pandas as pd

# Load CSV with correct separator
df = pd.read_csv("PB_All_2000_2021.csv", sep=";")

# Create a new target column: 1 = Safe, 0 = Unsafe
df["WaterQuality"] = df.apply(
    lambda row: 1 if (row["O2"] > 6 and row["Suspended"] < 50 and row["NH4"] < 0.5) else 0,
    axis=1
)

# Show first few rows with new column
print(df[["O2", "Suspended", "NH4", "WaterQuality"]].head())


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Select features and target
X = df[["O2", "Suspended", "NH4", "NO3", "NO2", "SO4", "PO4", "CL", "BSK5"]]  # Features
y = df["WaterQuality"]  # Target

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


import joblib

# Save the trained model to a file
joblib.dump(model, "water_quality_model.pkl")
print("Model saved as 'water_quality_model.pkl'")

