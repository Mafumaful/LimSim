import sqlite3
import pandas as pd
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# 1. Load Data from SQLite Database
PATH = "/Users/miakho/Code/LimSim/detector.db"
conn = sqlite3.connect(PATH)

# Load attack_stats
attack_query = "SELECT * FROM attack_stats"
attack_df = pd.read_sql_query(attack_query, conn)
attack_df.rename(columns={attack_df.columns[0]: 'time_step', attack_df.columns[1]: 'attack_type'}, inplace=True)

# Load cost_data
cost_query = "SELECT * FROM cost_data"
cost_df = pd.read_sql_query(cost_query, conn)
cost_df.rename(columns={
    cost_df.columns[0]: 'time_step',
    cost_df.columns[1]: 'path_cost',
    cost_df.columns[2]: 'traffic_rule_cost',
    cost_df.columns[3]: 'collision_possibility_cost',
    cost_df.columns[4]: 'total_cost'
}, inplace=True)

# Load predict_traj
traj_query = "SELECT * FROM predict_traj"
traj_df = pd.read_sql_query(traj_query, conn)
traj_df.rename(columns={
    traj_df.columns[0]: 'time_step',
    traj_df.columns[1]: 'vehicle_id',
    traj_df.columns[2]: 'x_pos',
    traj_df.columns[3]: 'y_pos',
    traj_df.columns[4]: 'p_traj'
}, inplace=True)

conn.close()

# 2. Merge Dataframes
merged_df = pd.merge(attack_df, cost_df, on='time_step')
merged_df = pd.merge(merged_df, traj_df, on='time_step')

# 3. Handle 'None' Values
# Check for 'None' in object-type columns
for column in merged_df.columns:
    if merged_df[column].dtype == object:
        print(f"\nChecking column: {column}")
        none_count = merged_df[merged_df[column] == 'None'][column].count()
        print(f"Number of 'None' entries: {none_count}")
        if none_count > 0:
            # Replace 'None' with np.nan
            merged_df[column].replace('None', np.nan, inplace=True)

# Convert relevant columns to numeric, coercing errors to NaN
feature_columns_initial = ['path_cost', 'traffic_rule_cost', 'collision_possibility_cost', 'total_cost']
merged_df[feature_columns_initial] = merged_df[feature_columns_initial].apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN in initial feature columns
merged_df.dropna(subset=feature_columns_initial, inplace=True)

# 4. Feature Engineering: Extract Features from Trajectories
def extract_features(traj_json):
    """
    Extract features from the JSON-encoded trajectory.
    """
    try:
        traj = json.loads(traj_json)
    except json.JSONDecodeError:
        traj = []
    
    if not traj or len(traj) < 2:
        # Not enough points to compute features
        return {
            'avg_velocity': 0,
            'avg_acceleration': 0,
            'trajectory_length': 0,
            'direction_changes': 0
        }
    
    # Convert to NumPy array for easier computations
    traj = np.array(traj)
    
    # Compute differences between consecutive points
    deltas = np.diff(traj, axis=0)  # Shape: (n-1, 2)
    distances = np.linalg.norm(deltas, axis=1)
    
    # Average Velocity (assuming constant time steps)
    avg_velocity = np.mean(distances)
    
    # Compute accelerations (differences of velocities)
    if len(distances) < 2:
        avg_acceleration = 0
    else:
        accel = np.diff(distances)
        avg_acceleration = np.mean(np.abs(accel))
    
    # Total Trajectory Length
    trajectory_length = np.sum(distances)
    
    # Direction Changes: Compute angles between consecutive deltas
    if len(deltas) < 2:
        direction_changes = 0
    else:
        # Normalize deltas
        norm_deltas = deltas / np.linalg.norm(deltas, axis=1, keepdims=True)
        # Compute dot products
        dot_products = np.einsum('ij,ij->i', norm_deltas[:-1], norm_deltas[1:])
        # Clip values to avoid numerical issues
        dot_products = np.clip(dot_products, -1.0, 1.0)
        angles = np.arccos(dot_products)
        # Count significant direction changes (e.g., angle > 30 degrees)
        direction_changes = np.sum(angles > (30 * np.pi / 180))
    
    return {
        'avg_velocity': avg_velocity,
        'avg_acceleration': avg_acceleration,
        'trajectory_length': trajectory_length,
        'direction_changes': direction_changes
    }

# Apply feature extraction
feature_dict = merged_df['p_traj'].apply(extract_features)
feature_df = pd.DataFrame(feature_dict.tolist())
merged_df = pd.concat([merged_df, feature_df], axis=1)

# 5. Finalize Feature Set
# Define all feature columns
feature_columns = [
    'path_cost',
    'traffic_rule_cost',
    'collision_possibility_cost',
    'total_cost',
    'avg_velocity',
    'avg_acceleration',
    'trajectory_length',
    'direction_changes'
]

# Ensure all feature columns are numeric
merged_df[feature_columns] = merged_df[feature_columns].apply(pd.to_numeric, errors='coerce')

# Handle any new NaN values resulting from conversion
imputer = SimpleImputer(strategy='mean')
merged_df[feature_columns] = imputer.fit_transform(merged_df[feature_columns])

# 6. Label Encoding
label_encoder = LabelEncoder()
merged_df['attack_type_encoded'] = label_encoder.fit_transform(merged_df['attack_type'])

print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# 7. Feature Scaling
scaler = StandardScaler()
merged_df[feature_columns] = scaler.fit_transform(merged_df[feature_columns])

# 8. Split Data into Training and Testing Sets
X = merged_df[feature_columns].values
y = merged_df['attack_type_encoded'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 9. Create PyTorch Datasets and DataLoaders
class AttackDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)  # For classification

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create dataset instances
train_dataset = AttackDataset(X_train, y_train)
test_dataset = AttackDataset(X_test, y_test)

# Create DataLoaders
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 10. Define the PyTorch Model
class AttackDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AttackDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

# Define model parameters
input_size = len(feature_columns)  # Number of features
hidden_size = 64
num_classes = len(label_encoder.classes_)   # Number of attack types

# Initialize the model
model = AttackDetector(input_size, hidden_size, num_classes)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

print(model)

# 11. Train the Model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Accumulate loss
        running_loss += loss.item() * batch_features.size(0)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total * 100
    
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# 12. Evaluate the Model
model.eval()  # Set model to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = model(batch_features)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

# Classification Report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# 13. Save the Model
torch.save(model.state_dict(), 'attack_detector.pth')
print("Model saved to attack_detector.pth")
