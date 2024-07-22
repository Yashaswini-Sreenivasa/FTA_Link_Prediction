import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

# Define the GraphSAGE model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Initialize GraphSAGE model
sage_model = GraphSAGE(train_data.num_features, 128, 64).to(device)
optimizer = optim.Adam(sage_model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.BCEWithLogitsLoss()

# RL Agent
class QLearningAgent:
    def __init__(self, n_actions, state_size, lr=0.01, gamma=0.99, epsilon=0.1):
        self.q_table = {}
        self.n_actions = n_actions
        self.state_size = state_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_key(self, state):
        return tuple(state)

    def get_q_value(self, state, action):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key][action]

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.n_actions))
        q_values = [self.get_q_value(state, a) for a in range(self.n_actions)]
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.n_actions)
        
        q_value = self.q_table[state_key][action]
        max_next_q_value = max(self.q_table[next_state_key])

        new_q_value = q_value + self.lr * (reward + self.gamma * max_next_q_value - q_value)
        self.q_table[state_key][action] = new_q_value

# Initialize RL agent
rl_agent = QLearningAgent(n_actions=2, state_size=64)

# Training loop
num_epochs = 150
for epoch in range(num_epochs):
    sage_model.train()
    optimizer.zero_grad()
    
    # GraphSAGE forward pass
    node_embeddings = sage_model(train_data.x.to(device), train_data.edge_index.to(device))
    
    # Negative sampling
    neg_edge_index = negative_sampling(edge_index=train_data.edge_index, num_nodes=train_data.num_nodes, num_neg_samples=train_data.edge_index.size(1))
    edge_index = torch.cat([train_data.edge_index, neg_edge_index], dim=1)
    edge_label = torch.cat([torch.ones(train_data.edge_index.size(1)), torch.zeros(neg_edge_index.size(1))])
    
    # RL edge prediction
    rewards = []
    for i in range(edge_index.size(1)):
        node_pair = node_embeddings[edge_index[:, i]].detach().cpu().numpy()
        state = np.mean(node_pair, axis=0)
        action = rl_agent.choose_action(state)
        
        if action == 1:  # Predict edge
            label = edge_label[i].item()
            reward = 1 if label == 1 else -1
        else:  # Do not predict edge
            reward = 0
        
        next_state = state  # In this example, next state is the same as current state
        rl_agent.update(state, action, reward, next_state)
        rewards.append(reward)
    
    # Compute loss for GraphSAGE
    edge_logits = (node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]).sum(dim=-1)
    loss = criterion(edge_logits, edge_label.to(device))
    
    loss.backward()
    optimizer.step()
    
    print(f'Epoch: {epoch+1}, Loss: {loss.item()}, Average Reward: {np.mean(rewards)}')

# Evaluate GraphSAGE model
sage_model.eval()
with torch.no_grad():
    node_embeddings = sage_model(train_data.x.to(device), train_data.edge_index.to(device))
    edge_logits = (node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]).sum(dim=-1)
    edge_probs = edge_logits.sigmoid().cpu().numpy()
    edge_labels = edge_label.cpu().numpy()
    auc_score = roc_auc_score(edge_labels, edge_probs)
    print(f'Test AUC: {auc_score:.4f}')

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(edge_labels, edge_probs)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Visualize the node embeddings using t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    node_embeddings_2d = tsne.fit_transform(node_embeddings.cpu().numpy())

    plt.figure()
    plt.scatter(node_embeddings_2d[:, 0], node_embeddings_2d[:, 1], c='blue', marker='o', s=10)
    plt.title('t-SNE visualization of node embeddings')
    plt.show()
