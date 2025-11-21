import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump

# --- 1. Load Data ---
# Replace 'agri_data.csv' with your actual file path
df = pd.read_csv('agri_data.csv') 

# Assuming your dataset has columns like:
# 'Product', 'Month', 'Year', 'Rainfall', 'WPI', 'Price' (Target)

# --- 2. Separate Features and Target ---
X = df.drop('Price', axis=1)
y = df['Price']

# --- 3. Preprocessing and Feature Engineering ---
# Define columns by type
categorical_features = ['Product', 'Month'] # Example categorical features
numerical_features = ['Year', 'Rainfall', 'WPI'] # Example numerical features

# Create preprocessing pipelines for different column types
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='passthrough'
)

# --- 4. Create and Train the Pipeline (Pre-processor + Model) ---
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

# --- 5. Evaluate (Optional, but good practice) ---
# from sklearn.metrics import r2_score
# y_pred = model.predict(X_test)
# print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# --- 6. Save the Trained Pipeline ---
dump(model, 'regression_model_pipeline.joblib')
print("Model pipeline saved as regression_model_pipeline.joblib")