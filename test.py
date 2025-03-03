import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("Starting data loading and preprocessing...")

# Load and prepare data
df = pd.read_csv('traces/cifar10/placements-aws.csv', 
                 names=['placement', 'local_bsz', 'step_time', 'sync_time'],
                 skiprows=1)  # Skip the first row

print(f"Data loaded. Shape: {df.shape}")
print(f"First few rows:\n{df.head()}")

# Convert local_bsz to integer
df['local_bsz'] = df['local_bsz'].astype(int)
df['step_time'] = df['step_time'].astype(float)
df['sync_time'] = df['sync_time'].astype(float)

# Convert placement to string first to ensure it's iterable
df['placement'] = df['placement'].astype(str)

# Add num_replicas (sum of all numbers in placement)
df['num_replicas'] = df['placement'].apply(lambda x: sum(int(char) for char in x if char.isdigit()))

# Add num_nodes (length of placement string)
df['num_nodes'] = df['placement'].apply(len)

print("Feature engineering completed.")
print(f"Data with new features:\n{df.head()}")

# Prepare features and target
X = df[['local_bsz', 'num_nodes', 'num_replicas']].values
y = df['step_time'].values

print(f"Feature matrix shape: {X.shape}, Target vector shape: {y.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling completed.")

# Define the kernel for Gaussian Process Regression
print("Defining Gaussian Process model...")
kernel = ConstantKernel() * RBF(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-5, 1e5)) + WhiteKernel(noise_level=1.0)

# Create and train the Gaussian Process Regression model
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=10, random_state=42)
print("Starting model training...")
gpr.fit(X_train_scaled, y_train)
print("Model training completed.")

# Make predictions
print("Making predictions...")
y_pred_train = gpr.predict(X_train_scaled)
y_pred_test = gpr.predict(X_test_scaled)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Training MSE: {train_mse:.4f}")
print(f"Testing MSE: {test_mse:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")
print(f"Optimized kernel parameters: {gpr.kernel_}")

# Function to predict step time given local_bsz, num_nodes, and num_replicas
def predict_step_time(local_bsz, num_nodes, num_replicas):
    features = np.array([[local_bsz, num_nodes, num_replicas]])
    features_scaled = scaler.transform(features)
    prediction, std = gpr.predict(features_scaled, return_std=True)
    return prediction[0], std[0]

# Example usage
print("Testing prediction function...")
example_local_bsz = 32
example_num_nodes = 4
example_num_replicas = 8
predicted_time, uncertainty = predict_step_time(example_local_bsz, example_num_nodes, example_num_replicas)
print(f"\nPredicted step time for local_bsz={example_local_bsz}, num_nodes={example_num_nodes}, num_replicas={example_num_replicas}: {predicted_time:.4f} ± {uncertainty:.4f}")

# Visualize the model's predictions vs actual values
print("Creating visualization plots...")
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Step Time')
plt.ylabel('Predicted Step Time')
plt.title('Gaussian Process Regression: Actual vs Predicted Step Time')
plt.grid(True)
plt.savefig('gpr_prediction_performance.png')
plt.close()
print("Prediction performance plot saved.")

# Feature importance visualization (based on kernel length scales)
if hasattr(gpr.kernel_, 'k1') and hasattr(gpr.kernel_.k1, 'k2') and hasattr(gpr.kernel_.k1.k2, 'length_scale'):
    length_scales = gpr.kernel_.k1.k2.length_scale
    feature_names = ['Local Batch Size', 'Number of Nodes', 'Number of Replicas']
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, 1/length_scales)
    plt.xlabel('Features')
    plt.ylabel('Relative Importance (1/length_scale)')
    plt.title('Feature Importance in Gaussian Process Model')
    plt.savefig('gpr_feature_importance.png')
    plt.close()
    print("Feature importance plot saved.")

print("Script execution completed successfully!")

# Now plot prediction performance for the analytical model
print("Creating prediction performance plot for analytical model...")

params_epoch2 = {
    'alpha_c': 0.020427763111669755,
    'beta_c': 0.0006846836959649913,
    'alpha_n': 0.01456101437196375,
    'beta_n': 0.007176994242966395,
    'alpha_r': 0.0682931949541188,
    'beta_r': 0.0006950884967738235,
    'gamma': 1.2792475977722224
}

def pred_compute_time(params, bsz):
    return params['alpha_c'] + params['beta_c'] * bsz

def pred_sync_time(params, num_nodes, num_replicas):
    if num_replicas == 1:
        return 0
    if num_nodes == 1:
        return params['alpha_r'] + params['beta_r'] * num_replicas
    else:
        return params['alpha_n'] + params['beta_n'] * num_replicas
    
def pred_step_time(params, num_nodes, num_replicas, bsz):
    compute_time = pred_compute_time(params, bsz)
    sync_time = pred_sync_time(params, num_nodes, num_replicas)
    return (compute_time ** params['gamma'] + sync_time ** params['gamma']) ** (1 / params['gamma'])

# Generate predictions using the analytical model
analytical_predictions = []
for i in range(len(X_test)):
    local_bsz = X_test[i][0]
    num_nodes = X_test[i][1]
    num_replicas = X_test[i][2]
    prediction = pred_step_time(params_epoch2, num_nodes, num_replicas, local_bsz)
    analytical_predictions.append(prediction)

# Calculate metrics for analytical model
analytical_mse = mean_squared_error(y_test, analytical_predictions)
analytical_r2 = r2_score(y_test, analytical_predictions)
print(f"Analytical Model MSE: {analytical_mse:.4f}")
print(f"Analytical Model R²: {analytical_r2:.4f}")

# Plot prediction performance for analytical model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, analytical_predictions, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Step Time')
plt.ylabel('Predicted Step Time')
plt.title('Analytical Model: Actual vs Predicted Step Time')
plt.grid(True)
plt.savefig('analytical_prediction_performance.png')
plt.close()
print("Analytical model prediction performance plot saved.")

# Compare both models in a single plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_test, alpha=0.7, label='GPR Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Step Time')
plt.ylabel('Predicted Step Time')
plt.title('GPR Model')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test, analytical_predictions, alpha=0.7, label='Analytical Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual Step Time')
plt.ylabel('Predicted Step Time')
plt.title('Analytical Model')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()
print("Model comparison plot saved.")