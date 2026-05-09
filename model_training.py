import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load dataset
# Pro tip: Use a raw string (r"") or forward slashes for Windows paths to avoid Unicode errors
path = r"C:\Users\Abhay\Downloads\fruaddetection_project\AIML Dataset.csv"
data = pd.read_csv(path)

# 2. Take smaller sample for fast training
data = data.sample(n=100000, random_state=42)

# 3. Drop unwanted columns (These don't help the math much)
data = data.drop(['nameOrig', 'nameDest'], axis=1)

# 4. Encode transaction type
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

# 5. Separate features and target
X = data.drop('isFraud', axis=1)
y = data['isFraud']

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Train model
# Added n_jobs=-1 to use all your CPU cores (faster!)
model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# 8. Quick Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# 9. Save model and the encoder (You'll need the encoder for the web app!)
pickle.dump(model, open("fraud_model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("Model and Encoder saved successfully!")