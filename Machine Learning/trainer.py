import sqlite3
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load and Merge Data
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

conn.close()

# Merge on time_step
merged_df = pd.merge(attack_df, cost_df, on='time_step')

# 2. Preprocessing
# Encode labels
label_encoder = LabelEncoder()
merged_df['attack_type_encoded'] = label_encoder.fit_transform(merged_df['attack_type'])
label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

# Feature scaling
feature_columns = ['path_cost', 'traffic_rule_cost', 'collision_possibility_cost', 'total_cost']
scaler = StandardScaler()
merged_df[feature_columns] = scaler.fit_transform(merged_df[feature_columns])

# 3. Split Data
X = merged_df[feature_columns].values
y = merged_df['attack_type_encoded'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 4. Create Datasets and DataLoaders
class AttackDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = AttackDataset(X_train, y_train)
test_dataset = AttackDataset(X_test, y_test)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 5. Define Model
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

input_size = len(feature_columns)
hidden_size = 64
num_classes = len(label_mapping)
model = AttackDetector(input_size, hidden_size, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(model)

# 6. Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 7. Train the Model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_features, batch_labels in train_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()
    
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total * 100
    
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# 8. Evaluate the Model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        outputs = model(batch_features)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=label_mapping.keys()))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# 9. Save the Model
torch.save(model.state_dict(), 'attack_detector.pth')
print("Model saved to attack_detector.pth")
