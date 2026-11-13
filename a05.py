
# Project A05: Noise model for one airfoil size only

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# STEP 1: Load the data file
df = pd.read_csv("AirfoilSelfNoise.csv")
print("Rows and columns:", df.shape)

# STEP 2: Clean the data
import numpy as np

# Keep only the biggest airfoil (chord 0.3048 m = 12 inches)
df = df[df["c"] == 0.3048].copy()
df["log_f"] = np.log10(df["f"])
df = df.dropna(subset=["log_f", "alpha", "U_infinity", "SSPL"])

# STEP 3: Choose inputs X and output y
inputs = ["log_f", "alpha", "U_infinity"]
X = df[inputs]
y = df["SSPL"]

# STEP 4: Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# STEP 5: Train
model = LinearRegression()
model.fit(X_train, y_train)

# STEP 6: Test
y_pred = model.predict(X_test)

print("R2 score     :", round(r2_score(y_test, y_pred), 3))
print("Average error:", round(mean_absolute_error(y_test, y_pred), 2), "dB")

print("Effect of each input (slope):")
for name, m in zip(inputs, model.coef_):
    print(" ", name, "->", round(m, 4))

# STEP 7: Predict a new case and draw Actual vs Predicted
new = pd.DataFrame({
    "log_f": [3.3], "alpha": [4.0], "U_infinity": [55.5]
})

print("Predicted noise for the 0.3048 m airfoil:",
      round(model.predict(new)[0], 2), "dB")

plt.scatter(y_test, y_pred, s=12, alpha=0.6)
low, high = y_test.min(), y_test.max()
plt.plot([low, high], [low, high],
         color="red", label="Perfect prediction")

plt.xlabel("Actual noise (dB)")
plt.ylabel("Predicted noise (dB)")
plt.title("Noise model for one airfoil size only")
plt.legend()
plt.show()
