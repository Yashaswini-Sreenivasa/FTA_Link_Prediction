import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np

# Check for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define a function to load fault tree data
def load_fault_tree(edge_csv_path, node_csv_path):
    edges_df = pd.read_csv(edge_csv_path)
    nodes_df = pd.read_csv(node_csv_path)

    # Fill missing values with 0
    nodes_df = nodes_df.fillna(0)

    # Extract node features
    node_features = nodes_df[['node_type', 'depth', 'gate_type', 'failure_probability', 'in_degree', 'out_degree', 'centrality']].values
    node_features = torch.tensor(node_features, dtype=torch.float)

    # Check for valid edge indices
    max_node_id = node_features.size(0) - 1
    valid_edges = edges_df[(edges_df['Source'] <= max_node_id) & (edges_df['Target'] <= max_node_id)]

    if len(valid_edges) < len(edges_df):
        print(f"Removed {len(edges_df) - len(valid_edges)} invalid edges")

    edge_index = torch.tensor(valid_edges.values.T, dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index)
    return data

# Paths to the large fault tree CSV files
edge_csv_path = r'C:\Users\yasha\src\data\Edges.csv'
node_csv_path = r'C:\Users\yasha\src\data\Nodes.csv'

# Load the fault tree data
fault_tree_data = load_fault_tree(edge_csv_path, node_csv_path)

# Print data statistics
print(f"Number of nodes: {fault_tree_data.num_nodes}")
print(f"Number of edges: {fault_tree_data.num_edges}")

# Define the transformation and split ratios
transform = RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=False, add_negative_train_samples=False)

try:
    # Apply the transformation
    data_transformed = transform(fault_tree_data)
    train_data, val_data, test_data = data_transformed
except ValueError as e:
    print(f"Error during data transformation: {e}")
    exit(1)

# Check the number of edges in the training data after transformation
num_edges_train = train_data.edge_index.size(1)
num_edges_val = val_data.edge_index.size(1)
num_edges_test = test_data.edge_index.size(1)

print(f"Number of edges in training set: {num_edges_train}")
print(f"Number of edges in validation set: {num_edges_val}")
print(f"Number of edges in test set: {num_edges_test}")

# Ensure sufficient edges for training
if num_edges_train < 100:
    raise ValueError(f"Insufficient number of edges for training: {num_edges_train}")

# Check for invalid edge indices
def check_edge_indices(data, num_nodes):
    invalid_edges = torch.where((data.edge_index >= num_nodes).any(dim=0))[0]
    if invalid_edges.size(0) > 0:
        print(f"Invalid edges found: {invalid_edges}")
        print(data.edge_index[:, invalid_edges])
        return False
    return True

if not check_edge_indices(train_data, fault_tree_data.num_nodes):
    raise ValueError("Invalid edge indices in training data")
if not check_edge_indices(val_data, fault_tree_data.num_nodes):
    raise ValueError("Invalid edge indices in validation data")
if not check_edge_indices(test_data, fault_tree_data.num_nodes):
    raise ValueError("Invalid edge indices in test data")

# Define the GraphSAGE model
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

model = Net(train_data.num_features, 128, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.BCEWithLogitsLoss()

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    z = model(train_data.x, train_data.edge_index)

    neg_edge_index = negative_sampling(edge_index=train_data.edge_index,
                                       num_nodes=train_data.num_nodes,
                                       num_neg_samples=train_data.edge_label_index.size(1),
                                       method='sparse')

    edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
    edge_label = torch.cat([train_data.edge_label, torch.zeros(neg_edge_index.size(1), dtype=torch.float)], dim=0)

    out = (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)
    loss = criterion(out, edge_label)
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
@torch.no_grad()
def test(data):
    model.eval()
    z = model(data.x, data.edge_index)
    out = (z[data.edge_label_index[0]] * z[data.edge_label_index[1]]).sum(dim=-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())

# Lists to store metrics for plotting
losses = []
val_aucs = []
test_aucs = []

# Training loop
best_val_auc = final_test_auc = 0
for epoch in range(1, 101):
    loss = train()
    val_auc = test(val_data)
    test_auc = test(test_data)

    # Store metrics
    losses.append(loss)
    val_aucs.append(val_auc)
    test_aucs.append(test_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        final_test_auc = test_auc
        # Save the best model if needed
        torch.save(model.state_dict(), 'model.pth')

    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_auc:.4f}, Test: {test_auc:.4f}')

print(f'Final Test AUC: {final_test_auc:.4f}')

# Plotting the metrics
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(val_aucs, label='Validation AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(test_aucs, label='Test AUC')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.legend()

plt.tight_layout()
plt.show()

# Additional Evaluation Metrics
def evaluate_model(data):
    model.eval()
    z = model(data.x, data.edge_index)
    out = (z[data.edge_label_index[0]] * z[data.edge_label_index[1]]).sum(dim=-1).sigmoid()
    y_true = data.edge_label.cpu().numpy()
    y_pred = out.detach().cpu().numpy()  # Detach tensor from computation graph
    
    # Confusion Matrix
    y_pred_binary = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)
    plt.figure()
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()
    
    # PCA for Node Embeddings
    pca = PCA(n_components=2)
    node_embeddings = z.cpu().detach().numpy()
    pca_result = pca.fit_transform(node_embeddings)
    plt.figure()
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data.x.cpu().numpy()[:, 0], cmap='viridis', s=10)
    plt.colorbar()
    plt.title('PCA of Node Embeddings')
    plt.show()

# Evaluate on validation and test datasets
print("Validation Metrics:")
evaluate_model(val_data)

print("Test Metrics:")
evaluate_model(test_data)
