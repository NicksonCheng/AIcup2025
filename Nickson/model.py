# model.py
import torch
import torch.nn as nn

class TransformerStockPredictor(nn.Module):
    def __init__(self, input_dim, d_model=512, n_heads=8, n_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        A powerful Transformer-based stock predictor optimized for high-memory hardware.
        
        Args:
            input_dim (int): Number of input features (e.g., 50 from feature selection).
            d_model (int): Dimension of the model (increased to 512 for richer representations).
            n_heads (int): Number of attention heads (8 for multi-head attention).
            n_layers (int): Number of transformer layers (6 for deeper processing).
            dim_feedforward (int): Size of the feedforward network (2048 for more capacity).
            dropout (float): Dropout rate (0.1 for regularization).
        """
        super(TransformerStockPredictor, self).__init__()
        
        # Input projection to a larger embedding space
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Transformer encoder with increased capacity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Use batch_first=True for easier shape handling
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layer
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
        
        Returns:
            torch.Tensor: Predictions of shape (batch_size, 1)
        """
        # Project input features to d_model dimension
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Pool across the sequence dimension (mean pooling)
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Final prediction
        x = self.fc(x)  # (batch_size, 1)
        return self.sigmoid(x)

# Example instantiation for testing
if __name__ == "__main__":
    # Test with dummy data
    model = TransformerStockPredictor(input_dim=50)
    x = torch.randn(32, 20, 50)  # batch_size=32, seq_len=20, input_dim=50
    output = model(x)
    print(f"Output shape: {output.shape}")  # Should be (32, 1)