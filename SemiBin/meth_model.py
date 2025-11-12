import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MethylationAttentionLayer(nn.Module):
    """Attention layer specifically for methylation features"""
    def __init__(self, n_motifs, hidden_dim=64, n_heads=4):
        super().__init__()

        self.n_motifs = n_motifs

        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )
        
        self.value_projection = nn.Linear(1, hidden_dim)

        # Positional encoding for motif positions
        self.position_embedding = nn.Parameter(torch.randn(1, n_motifs, hidden_dim))

        # Positional encoding for motif positions
        self.absent_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, methylation_values, motif_present_mask):
        """
        methylation_values: (batch_size, n_motifs) - methylation percentages
        motif_present_mask: (batch_size, n_motifs) - binary mask, 1 if motif is present, 0 if absent
        """
        batch_size = methylation_values.shape[0]
        
        # Handle NaN values by creating a mask
        motif_present_mask = motif_present_mask.bool()
        methylation_values = torch.nan_to_num(methylation_values, 0.0)
        
        # Reshape for projection: (batch, n_motifs, 1)
        meth_values_expanded = methylation_values.unsqueeze(-1)
        value_embeddings = self.value_projection(meth_values_expanded)
        
        # Add positional information
        embeddings_with_position = value_embeddings + self.position_embedding

        absent_embedding_expanded = self.absent_embedding.expand(batch_size, self.n_motifs, -1)
        
        combined_embeddings = torch.where(
            motif_present_mask.unsqueeze(-1),
            embeddings_with_position,
            absent_embedding_expanded
        )

        attended, attended_weights = self.attention(
            combined_embeddings,
            combined_embeddings,
            combined_embeddings
        )

        pooled = attended.mean(dim = 1)
        
        return self.output_projection(pooled)


class Self_encoding_dual_stream(nn.Module):
    """
    Enhanced model that handles kmer and methylation features separately
    This replaces Self_encoding_single or Self_encoding_multiple
    """
    def __init__(self, total_features, n_kmer_features, n_motif_features, n_motif_present_features):
        super().__init__()

        assert n_motif_features == n_motif_present_features, f"Mismatch: {n_motif_features} motif features but {n_motif_present_features} presence indicators"
        
        # Store dimensions for later use
        self.total_features = total_features
        self.n_kmer_features = n_kmer_features
        self.n_motif_features = n_motif_features
        self.n_motif_present_features = n_motif_present_features
        
        # Kmer encoder (your existing architecture)
        self.kmer_encoder = nn.Sequential(
            nn.Linear(n_kmer_features, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        
        self.methylation_attention = MethylationAttentionLayer(
            n_motifs = n_motif_features,
            hidden_dim = 64,
        )
        
        # Methylation encoder with attention
        self.methylation_encoder = nn.Sequential(
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
        )
        
        # Learnable weights for combining features
        self.feature_weights = nn.Parameter(torch.ones(2))
        
        # Final encoder (combines both streams)
        self.final_encoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 100),  # Your original embedding dimension
        )
        
        # Decoder (reconstructs the original concatenated features)
        self.decoder1 = nn.Sequential(
            nn.Linear(100, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, total_features),
            nn.Sigmoid(),
        )
        
    def split_features(self, x):
        """Split concatenated features into kmer and methylation"""
        kmer_end = 136
        motif_end = kmer_end + self.n_motif_features
        motif_present_end = motif_end + self.n_motif_present_features
        
        kmer_features = x[:, :kmer_end]
        methylation_values = x[:, kmer_end:motif_end]
        motif_present_mask = x[:, motif_end:motif_present_end]
           
        return kmer_features, methylation_values, motif_present_mask
    
    def encode_features(self, x):
        """Encode input features to embedding"""
        kmer_feats, meth_values, meth_mask = self.split_features(x)
        
        # Encode each stream
        kmer_embed = self.kmer_encoder(kmer_feats)

        # Encode methylation stream with attention and mask
        meth_attention_out = self.methylation_attention(meth_values, meth_mask)
        meth_embed = self.methylation_encoder(meth_attention_out)
        
        # Normalize before combining
        kmer_embed = F.normalize(kmer_embed, dim=-1)
        meth_embed = F.normalize(meth_embed, dim=-1)
        
        # Weighted combination
        weights = F.softmax(self.feature_weights, dim=0)
        combined = weights[0] * kmer_embed + weights[1] * meth_embed
        
        # Final encoding
        embedding = self.final_encoder(combined)
        return embedding
    
    def forward(self, input1, input2):
        """Forward pass for training with pairs"""
        return self.encode_features(input1), self.encode_features(input2)
    
    def decoder(self, input1, input2):
        """Decoder for reconstruction loss"""
        return self.decoder1(input1), self.decoder1(input2)
    
    def embedding(self, input):
        """Get embedding for a single input (for inference)"""
        return self.encode_features(input)
    
    def get_feature_weights(self):
        """Get the learned importance of each feature type"""
        weights = F.softmax(self.feature_weights, dim=0)
        return {'kmer': weights[0].item(), 'methylation': weights[1].item()}
    
    def save_with_params_to(self, path):
        torch.save({
            'model_name': 'Self_encoding_dual_stream',
            'model_state_dict': self.state_dict(),
            'params': [self.total_features, self.n_kmer_features, self.n_motif_features, self.n_motif_present_features],
        }, path)


# Modified loss function that works with your existing setup
def loss_function(embedding1, embedding2, label):
    relu = torch.nn.ReLU()
    
    # Euclidean distance between embeddings
    d = torch.norm(embedding1 - embedding2, p=2, dim=1)
    
    # Contrastive loss
    square_pred = torch.square(d)  # For must-link pairs (label=1)
    margin_square = torch.square(relu(1 - d))  # For cannot-link pairs (label=0)
    
    supervised_loss = torch.mean(
        label * square_pred + (1 - label) * margin_square
    )
    
    return supervised_loss


