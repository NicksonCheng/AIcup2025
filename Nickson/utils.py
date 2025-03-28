# utils.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import os

class StockDataset(Dataset):
    def __init__(self, X, y=None, seq_len=20):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        start_idx = max(0, idx - self.seq_len + 1)
        seq = self.X[start_idx:idx+1]
        if len(seq) < self.seq_len:
            padding = torch.zeros(self.seq_len - len(seq), seq.shape[1])
            seq = torch.cat([padding, seq], dim=0)
        if self.y is not None:
            return seq, self.y[idx]
        return seq

def batch_smote(X, y, batch_size=10000):
    n_samples = len(y)
    for start_idx in tqdm(range(0, n_samples, batch_size), desc="Applying SMOTE"):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        
        # Count minority class samples
        minority_count = sum(y_batch)
        
        # Skip SMOTE if no minority samples or all samples are minority
        if minority_count == 0 or minority_count == len(y_batch):
            yield X_batch, y_batch
            continue
        
        # Adjust n_neighbors based on minority count (minimum of 5 or available samples - 1)
        n_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1
        try:
            smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X_batch, y_batch)
            yield X_balanced, y_balanced
        except ValueError as e:
            print(f"SMOTE failed for batch {start_idx}-{end_idx} with minority count {minority_count}: {e}", flush=True)
            yield X_batch, y_batch  # Fallback to original batch if SMOTE fails

def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, save_path="trained_model.pth"):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Training", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_preds, val_true = [], []
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_preds.extend(output.squeeze().cpu().numpy())
                val_true.extend(y_batch.cpu().numpy())

        val_f1 = f1_score(val_true, (np.array(val_preds) > 0.5).astype(int))
        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val F1: {val_f1:.4f}", flush=True)
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}", flush=True)
    return model

def load_model(model_class, input_dim, model_path="trained_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}", flush=True)
    return model

def predict(model, X_test, seq_len=20, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    dataset = StockDataset(X_test, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions = []

    with torch.no_grad():
        for X_batch in tqdm(loader, desc="Predicting"):
            X_batch = X_batch.to(device)
            output = model(X_batch)
            preds = (output.squeeze().cpu().numpy() > 0.5).astype(int)
            if preds.ndim == 0:
                predictions.append(preds.item())
            else:
                predictions.extend(preds.tolist())

    return predictions