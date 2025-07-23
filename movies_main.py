import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from joblib import dump


# Load the dataset
df = pd.read_csv(r"C:\Users\spand\Downloads\Predicting Movie Success\movie_metadata.csv")

# Encode all object (string) columns to numeric using LabelEncoder
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # optional, for inverse_transform later

# Convert imdb_score to numeric and drop rows with missing/invalid values
df['imdb_score'] = pd.to_numeric(df['imdb_score'], errors='coerce')
df = df.dropna(subset=['imdb_score'])

# Bin imdb_score to classify movies
bins = [1, 3, 6, 10]
labels = ['Flop', 'Average', 'Hit']
df['Classify'] = pd.cut(df['imdb_score'], bins=bins, labels=labels)

# Drop rows with missing Classify (outside bins)
df = df.dropna(subset=['Classify'])

# Encode Classify to numeric labels
class_le = LabelEncoder()
df['Classify'] = class_le.fit_transform(df['Classify'].astype(str))

# Ensure all features are numeric (very important)
X = df.drop(columns=['imdb_score', 'Classify'])
X = X.apply(pd.to_numeric, errors='coerce')  # force numeric conversion
X = X.dropna(axis=1, how='any')              # drop columns with any non-numeric data

y = df['Classify']

# Train initial model to get feature importances
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X, y)

# Get top 8 features
importances = rfc.feature_importances_
feature_imp = pd.Series(importances, index=X.columns, name='importance')
top_features = feature_imp.sort_values(ascending=False).head(8).index.tolist()

# Save top features
with open("selected_features.txt", "w") as f:
    for feature in top_features:
        f.write(f"{feature}\n")

# Retrain model with top features only
X_top = X[top_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_top)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

# Save the trained model to a file
dump(rfc, 'movie_success_model.joblib')