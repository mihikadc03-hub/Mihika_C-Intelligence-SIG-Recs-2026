import sys
import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, auc, log_loss, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add dataset directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')))
# pyrefly: ignore [missing-import]
from data_utils import load_data

torch.manual_seed(42)
np.random.seed(42)

def resolve_data_dir(data_dir_arg=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    if data_dir_arg:
        candidates.append(data_dir_arg)
        candidates.append(os.path.join(script_dir, data_dir_arg))
    candidates.extend([
        os.path.abspath(os.path.join(script_dir, '..', '..', 'datasets', 'ctr_data')),
        os.path.abspath(os.path.join(os.getcwd(), 'mandatory-tasks', 'recommender-systems-cf-to-dlrm', 'datasets', 'ctr_data')),
        os.path.abspath(os.path.join(os.getcwd(), 'datasets', 'ctr_data')),
    ])
    for p in candidates:
        if p and os.path.isdir(p) and os.path.exists(os.path.join(p, 'train.csv')):
            return p
    # Try extracting zip if not unzipped
    zip_path = os.path.abspath(os.path.join(script_dir, '..', '..', 'datasets', 'dataset (tasks 2 and 3).zip'))
    if os.path.exists(zip_path):
        import zipfile
        target_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'datasets', 'ctr_data'))
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        return target_dir
    return os.path.abspath(os.path.join(script_dir, '..', '..', 'datasets', 'ctr_data'))

class VanillaNN(nn.Module):
    def __init__(self, num_dense_features, vocab_sizes, embed_dim=16, hidden_dims=[256, 128, 64]):
        super(VanillaNN, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in vocab_sizes
        ])
        
        # Input dim: dense + concatenated embeddings
        input_dim = num_dense_features + len(vocab_sizes) * embed_dim
        
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            input_dim = dim
            
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, dense_x, sparse_x):
        # Embed categorical features
        emb_list = [emb(sparse_x[:, i]) for i, emb in enumerate(self.embeddings)]
        emb_cat = torch.cat(emb_list, dim=1)
        
        # Concatenate dense and sparse features
        x = torch.cat([dense_x, emb_cat], dim=1)
        
        return self.mlp(x)

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=3):
        super(CrossNetwork, self).__init__()
        self.num_layers = num_layers
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim, 1) * 0.01) for _ in range(num_layers)
        ])
        self.cross_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim, 1)) for _ in range(num_layers)
        ])
        
    def forward(self, x0):
        # x0: (batch_size, input_dim)
        x_l = x0.unsqueeze(2) # (batch_size, input_dim, 1)
        x0_expanded = x0.unsqueeze(2)
        
        for w, b in zip(self.cross_weights, self.cross_biases):
            # x_l.transpose(1,2) is (batch_size, 1, input_dim)
            # w is (input_dim, 1)
            # w.unsqueeze(0) is (1, input_dim, 1)
            
            # Efficient calculation: x_l^T * w is scalar per batch
            # torch.bmm(x_l.transpose(1, 2), w.expand(batch_size, -1, -1))
            
            w_exp = w.expand(x_l.size(0), -1, -1)
            b_exp = b.expand(x_l.size(0), -1, -1)
            
            dot_prod = torch.bmm(x_l.transpose(1, 2), w_exp) # (batch_size, 1, 1)
            
            # x_{l+1} = x_0 * dot_prod + b + x_l
            x_l = x0_expanded * dot_prod + b_exp + x_l
            
        return x_l.squeeze(2)

class DeepCrossNetwork(nn.Module):
    def __init__(self, num_dense_features, vocab_sizes, embed_dim=16, hidden_dims=[256, 128], num_cross_layers=3):
        super(DeepCrossNetwork, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in vocab_sizes
        ])
        
        input_dim = num_dense_features + len(vocab_sizes) * embed_dim
        
        # Deep Network
        layers = []
        curr_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            curr_dim = dim
        self.deep_network = nn.Sequential(*layers)
        
        # Cross Network
        self.cross_network = CrossNetwork(input_dim, num_layers=num_cross_layers)
        
        # Final output layer
        self.out_layer = nn.Linear(input_dim + curr_dim, 1)
        
    def forward(self, dense_x, sparse_x):
        emb_list = [emb(sparse_x[:, i]) for i, emb in enumerate(self.embeddings)]
        emb_cat = torch.cat(emb_list, dim=1)
        x0 = torch.cat([dense_x, emb_cat], dim=1)
        
        deep_out = self.deep_network(x0)
        cross_out = self.cross_network(x0)
        
        concat_out = torch.cat([cross_out, deep_out], dim=1)
        return self.out_layer(concat_out)

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for dense_x, sparse_x, y in train_loader:
            dense_x, sparse_x, y = dense_x.to(device), sparse_x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(dense_x, sparse_x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(y)
            
        train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        val_loss, _, _ = evaluate_model(model, val_loader, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
    return train_losses, val_losses

def evaluate_model(model, data_loader, device='cpu'):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for dense_x, sparse_x, y in data_loader:
            dense_x, sparse_x, y = dense_x.to(device), sparse_x.to(device), y.to(device)
            out = model(dense_x, sparse_x)
            loss = criterion(out, y)
            
            total_loss += loss.item() * len(y)
            probs = torch.sigmoid(out).cpu().numpy()
            
            all_preds.extend(probs)
            all_targets.extend(y.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss, np.array(all_preds), np.array(all_targets)

def compute_metrics(y_true, y_pred_prob, threshold=0.5):
    y_pred = (y_pred_prob >= threshold).astype(int)
    roc_auc = roc_auc_score(y_true, y_pred_prob)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(recall, precision)
    loss = log_loss(y_true, y_pred_prob)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'ROC-AUC': roc_auc,
        'PR-AUC': pr_auc,
        'LogLoss': loss,
        'Accuracy': acc,
        'F1-Score': f1
    }

def main():
    parser = argparse.ArgumentParser(description="Task 2: Neural CTR Models")
    parser.add_argument('--data_dir', type=str, default=None, help='Path to unzipped CTR data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = resolve_data_dir(args.data_dir)
    print(f"Using dataset directory: {data_dir}")
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    print("Loading data...")
    train_dataset, val_dataset, test_dataset, preprocessor = load_data(train_path, test_path)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    num_dense = len(preprocessor.num_cols)
    vocab_sizes = preprocessor.vocab_sizes
    
    print("\n--- Vanilla Neural Network ---")
    vanilla_model = VanillaNN(num_dense, vocab_sizes).to(device)
    vanilla_train_loss, vanilla_val_loss = train_model(vanilla_model, train_loader, val_loader, epochs=args.epochs, device=device)
    
    print("\nEvaluating Vanilla NN on Test Set...")
    _, vanilla_preds, test_y = evaluate_model(vanilla_model, test_loader, device)
    vanilla_metrics = compute_metrics(test_y, vanilla_preds)
    print("Vanilla NN Metrics:", vanilla_metrics)
    
    print("\n--- Deep & Cross Network (DCN) ---")
    dcn_model = DeepCrossNetwork(num_dense, vocab_sizes).to(device)
    dcn_train_loss, dcn_val_loss = train_model(dcn_model, train_loader, val_loader, epochs=args.epochs, device=device)
    
    print("\nEvaluating DCN on Test Set...")
    _, dcn_preds, _ = evaluate_model(dcn_model, test_loader, device)
    dcn_metrics = compute_metrics(test_y, dcn_preds)
    print("DCN Metrics:", dcn_metrics)
    
    # Plotting Learning Curves
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(output_dir, 'learning_curves.png')
    
    epochs_range = list(range(1, len(vanilla_train_loss) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, vanilla_train_loss, marker='o', label='Vanilla NN Train')
    plt.plot(epochs_range, vanilla_val_loss, marker='o', label='Vanilla NN Val')
    plt.plot(epochs_range, dcn_train_loss, marker='s', linestyle='--', label='DCN Train')
    plt.plot(epochs_range, dcn_val_loss, marker='s', linestyle='--', label='DCN Val')
    plt.xlabel('Epochs')
    plt.ylabel('BCE Loss')
    plt.title('Training & Validation Curves')
    if len(epochs_range) <= 20:
        plt.xticks(epochs_range)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    print(f"\nPlots saved to {plot_path}")

if __name__ == '__main__':
    main()
