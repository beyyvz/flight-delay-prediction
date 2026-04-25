import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------------------
# Generate Synthetic Realistic Dataset
# -----------------------------------------
np.random.seed(42)
n = 200

data = pd.DataFrame({
    "DEPARTURE_TIME": np.random.randint(0, 2359, n),
    "DISTANCE": np.random.randint(100, 3000, n),
})

# Realistic delay rule:
# - Flights after 5pm more likely to be delayed
# - Long flights more likely to be delayed
data["DELAYED"] = (
    (data["DEPARTURE_TIME"] > 1700).astype(int) |
    (data["DISTANCE"] > 2000).astype(int)
)

# Add noise (10% flipped labels)
flip = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
data["DELAYED"] = data["DELAYED"] ^ flip

X = data[["DEPARTURE_TIME", "DISTANCE"]]
y = data["DELAYED"]

# -----------------------------------------
# Train/Test Split
# -----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------
# Train Logistic Regression Model
# -----------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------------------
# Evaluate Model
# -----------------------------------------
accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")

# -----------------------------------------
# Save Model
# -----------------------------------------
joblib.dump(model, "flight_delay_model.pkl")
print("Model saved as flight_delay_model.pkl")
