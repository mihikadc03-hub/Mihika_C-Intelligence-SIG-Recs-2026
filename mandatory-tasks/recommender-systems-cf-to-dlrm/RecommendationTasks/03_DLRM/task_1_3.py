import sys
import os
import argparse
import time

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

class DLRM(nn.Module):
    def __init__(self, num_dense_features, vocab_sizes, embed_dim=16, 
                 bottom_mlp_dims=[64, 16], top_mlp_dims=[128, 64],
                 use_interactions=True):
        super(DLRM, self).__init__()
        self.use_interactions = use_interactions
        self.embed_dim = embed_dim
        
        # Ensure the last layer of bottom MLP matches embedding dimension
        assert bottom_mlp_dims[-1] == embed_dim, "Last layer of bottom MLP must match embed_dim"
        
        # Categorical Embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim) for vocab_size in vocab_sizes
        ])
        
        # Bottom MLP for dense features
        bottom_layers = []
        curr_dim = num_dense_features
        for dim in bottom_mlp_dims:
            bottom_layers.append(nn.Linear(curr_dim, dim))
            bottom_layers.append(nn.ReLU())
            curr_dim = dim
        self.bottom_mlp = nn.Sequential(*bottom_layers)
        
        # Interaction layer output dimension
        num_cat = len(vocab_sizes)
        if use_interactions:
            # Pairwise interactions between all categorical embeddings + the dense representation
            # Number of vectors = num_cat + 1
            # Number of interactions = (num_cat + 1) * num_cat / 2
            interaction_dim = (num_cat + 1) * num_cat // 2
            top_input_dim = interaction_dim + embed_dim # Interactions + Dense Representation
        else:
            # Ablation: just concatenate all embeddings + dense representation
            top_input_dim = (num_cat + 1) * embed_dim
            
        # Top MLP
        top_layers = []
        curr_dim = top_input_dim
        for dim in top_mlp_dims:
            top_layers.append(nn.Linear(curr_dim, dim))
            top_layers.append(nn.ReLU())
            top_layers.append(nn.Dropout(0.1))
            curr_dim = dim
        top_layers.append(nn.Linear(curr_dim, 1))
        self.top_mlp = nn.Sequential(*top_layers)
        
    def interact_features(self, dense_emb, cat_embs):
        # cat_embs is a list of tensors of shape (B, embed_dim)
        # dense_emb is of shape (B, embed_dim)
        
        # Stack all embeddings: shape (B, num_cat + 1, embed_dim)
        all_embs = torch.stack([dense_emb] + cat_embs, dim=1)
        
        # Batch matrix multiplication: (B, num_features, embed_dim) x (B, embed_dim, num_features)
        # Result is (B, num_features, num_features) containing pairwise dot products
        interactions = torch.bmm(all_embs, all_embs.transpose(1, 2))
        
        # Extract upper triangular part (excluding diagonal)
        B, N, _ = interactions.shape
        triu_indices = torch.triu_indices(N, N, offset=1, device=interactions.device)
        
        interaction_flat = interactions[:, triu_indices[0], triu_indices[1]] # (B, N*(N-1)/2)
        
        # Concatenate dense representation with interactions
        return torch.cat([dense_emb, interaction_flat], dim=1)
        
    def forward(self, dense_x, sparse_x):
        # 1. Process dense features
        dense_emb = self.bottom_mlp(dense_x)
        
        # 2. Process categorical features
        cat_embs = [emb(sparse_x[:, i]) for i, emb in enumerate(self.embeddings)]
        
        # 3. Interactions
        if self.use_interactions:
            top_input = self.interact_features(dense_emb, cat_embs)
        else:
            # Ablation: no interaction, just concat
            top_input = torch.cat([dense_emb] + cat_embs, dim=1)
            
        # 4. Top MLP
        return self.top_mlp(top_input)

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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description="Task 3: DLRM")
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
    
    print("\n--- DLRM (With Interactions) ---")
    dlrm_model = DLRM(num_dense, vocab_sizes, use_interactions=True).to(device)
    print(f"Model Parameters: {count_parameters(dlrm_model):,}")
    
    start_time = time.time()
    dlrm_train_loss, dlrm_val_loss = train_model(dlrm_model, train_loader, val_loader, epochs=args.epochs, device=device)
    print(f"Training completed in {time.time() - start_time:.2f}s")
    
    print("\nEvaluating DLRM on Test Set...")
    _, dlrm_preds, test_y = evaluate_model(dlrm_model, test_loader, device)
    dlrm_metrics = compute_metrics(test_y, dlrm_preds)
    print("DLRM Metrics:", dlrm_metrics)
    
    print("\n--- Ablation: DLRM (No Interactions) ---")
    dlrm_no_int_model = DLRM(num_dense, vocab_sizes, use_interactions=False).to(device)
    print(f"Model Parameters: {count_parameters(dlrm_no_int_model):,}")
    
    ablation_train_loss, ablation_val_loss = train_model(dlrm_no_int_model, train_loader, val_loader, epochs=args.epochs, device=device)
    
    print("\nEvaluating Ablation Model on Test Set...")
    _, ablation_preds, _ = evaluate_model(dlrm_no_int_model, test_loader, device)
    ablation_metrics = compute_metrics(test_y, ablation_preds)
    print("Ablation Metrics:", ablation_metrics)
    
    # Plotting Learning Curves
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(output_dir, 'dlrm_learning_curves.png')
    
    epochs_range = list(range(1, len(dlrm_train_loss) + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, dlrm_train_loss, marker='o', label='DLRM Train')
    plt.plot(epochs_range, dlrm_val_loss, marker='o', label='DLRM Val')
    plt.plot(epochs_range, ablation_train_loss, marker='s', linestyle='--', label='No-Interaction Train')
    plt.plot(epochs_range, ablation_val_loss, marker='s', linestyle='--', label='No-Interaction Val')
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
