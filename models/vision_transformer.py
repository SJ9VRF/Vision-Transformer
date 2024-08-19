# This file contains the implementation of the Vision Transformer model.

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, query, key, value):
        attn_output, _ = self.attention(query, key, value)
        x = query + attn_output
        x = self.norm1(x)
        ff_output = self.feed_forward(x)
        output = x + ff_output
        output = self.norm2(output)
        return output

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.num_layers = config['num_layers']
        self.num_classes = config['num_classes']
        self.patch_size = config['patch_size']
        self.num_channels = config['num_channels']

        self.patch_embedding = nn.Conv2d(self.num_channels, self.embed_dim,
                                         kernel_size=self.patch_size, stride=self.patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn((32*32 // self.patch_size**2) + 1, self.embed_dim))

        self.transformer_blocks = nn.ModuleList([
            AttentionBlock(self.embed_dim, self.num_heads, config['dropout_rate'])
            for _ in range(self.num_layers)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embedding(x)  # Transform to patch embeddings
        x = x.flatten(2)
        x = x.transpose(1, 2)  # B, N, embed_dim

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # add cls token
        x += self.positional_embedding[:x.size(1), :]

        for block in self.transformer_blocks:
            x = block(x, x, x)

        cls_token_final = x[:, 0]  # Use the CLS token to classify
        x = self.mlp_head(cls_token_final)

        return x
