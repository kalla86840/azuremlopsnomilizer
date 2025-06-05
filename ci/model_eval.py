
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_test = joblib.load('X_test.joblib')
y_test = joblib.load('y_test.joblib')
normalizer = joblib.load('scaler.joblib')
model = joblib.load('model.joblib')

X_test_normalized = normalizer.transform(X_test)
y_pred = model.predict(X_test_normalized)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with open('metrics.txt', 'w') as f:
    f.write(f"MAE: {mae:.2f}\nMSE: {mse:.2f}\nR2: {r2:.2f}\n")
