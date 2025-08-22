import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .attention_modules import CustomTransformerEncoderLayer


class CustomVisionTransformer(nn.Module):
    """
    学習用のVision Transformer (ViT) 実装
    
    処理の流れ:
    1. Patch Embedding: 画像をパッチに分割してベクトルに変換
    2. Position Embedding: 各パッチの位置情報を追加
    3. CLS Token: 分類用の特別トークンを追加
    4. Transformer Encoder: Multi-head attentionで画像特徴を学習
    5. Final Representation: CLSトークンから画像レベルの特徴量を抽出
    
    CLIPでの役割:
    - 入力画像 [batch, 3, 224, 224] を埋め込みベクトル [batch, embed_dim] に変換
    - テキストエンコーダーと同じ次元空間で画像を表現
    """
    
    def __init__(self, img_size=224, patch_size=16, embed_dim=512, 
                 num_heads=8, num_layers=12, num_channels=3):
        super().__init__()
        
        # パラメータ設定
        self.img_size = img_size           # 入力画像サイズ (224x224)
        self.patch_size = patch_size       # パッチサイズ (16x16)  
        self.num_patches = (img_size // patch_size) ** 2  # パッチ数 (14x14=196)
        
        # Step 1: Patch Embedding Layer
        # 画像を小さなパッチに分割し、各パッチを埋め込みベクトルに変換
        # Conv2dを使って効率的にパッチ分割+線形変換を同時実行
        # 出力サイズ = (入力サイズ - カーネルサイズ + 2×パディング) / ストライド + 1
        # 出力H = (224 - 16 + 0) / 16 + 1 = 208/16 + 1 = 13 + 1 = 14
        # 出力W = (224 - 16 + 0) / 16 + 1 = 14
        self.patch_embed = nn.Conv2d(
            in_channels=num_channels,      # RGB: 3チャンネル
            out_channels=embed_dim,        # 埋め込み次元
            kernel_size=patch_size,        # 16x16のカーネル
            stride=patch_size,             # 16x16ずつ移動（重複なし）
            padding=0                      # パディングなし
        )
        # 出力: [batch, embed_dim, 14, 14] → flatten → [batch, 196, embed_dim]
        
        # Step 2: CLS Token (分類トークン：テキストにおけるEOTトークンのような役割)
        # 画像全体の情報を集約するための特別なトークン
        # BERTのCLSトークンと同様の役割
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Step 3: Position Embedding
        # 各パッチの空間的位置情報を学習可能な埋め込みで表現
        # パッチの位置関係をTransformerに教える
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)  # +1 for CLS token
        )
        
        # Step 4: Transformer Encoder Layers
        # 複数のTransformer層でパッチ間の関係を学習
        self.transformer_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(embed_dim, num_heads, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Step 5: Layer Normalization
        # 最終出力の正規化
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Step 6: Dropout (正則化)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, images):
        """
        Vision Transformerの順伝播
        
        Args:
            images: [batch_size, channels, height, width] - 入力画像
                   通常は [batch_size, 3, 224, 224] (RGB画像)
                   ImageNet標準化済み (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        
        Returns:
            image_features: [batch_size, embed_dim] - 画像レベルの特徴量
        """
        
        batch_size = images.shape[0]
        
        # Step 1: Patch Embedding
        # 画像をパッチに分割して埋め込みベクトルに変換
        # [batch, 3, 224, 224] → [batch, embed_dim, 14, 14]
        x = self.patch_embed(images)
        
        # Step 2: Flatten and Transpose
        # [batch, embed_dim, 14, 14] → [batch, 196, embed_dim](196個のパッチ＝文章なら196トークン)
        x = x.flatten(start_dim=2).transpose(1, 2)  # flatten(start_dim=2)
        
        # Step 3: CLS Token追加
        # 分類用トークンをシーケンスの先頭に追加
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, embed_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch, 197, embed_dim] (1 + 196)
        
        # Step 4: Position Embedding追加
        # 各パッチとCLSトークンに位置情報を付与
        x = x + self.pos_embedding  # Broadcasting: [batch, 197, embed_dim]
        x = self.dropout(x)
        
        # Step 5: Transformer Encoder通す
        # 各パッチ間の関係性を学習
        for layer in self.transformer_layers:
            x = layer(x)  # 画像にはpadding maskは不要
            # [batch, 197, embed_dim]
        
        # Step 6: CLS Token抽出
        # 先頭のCLSトークンが画像全体の特徴を集約（テキストのEOTトークンと同様）
        cls_output = x[:, 0]  # [batch_size, embed_dim]
        
        # Step 7: Final Layer Normalization
        # 最終特徴量を正規化
        image_features = self.ln_final(cls_output)
        
        return image_features
    


class ImageEncoder(nn.Module):
    """
    CLIPの画像エンコーダー
    
    選択可能なアーキテクチャ:
    1. Vision Transformer (ViT) - デフォルト
    2. ResNet - 従来のCNNベース
    
    役割:
    - 画像を固定長の埋め込みベクトルに変換
    - テキストエンコーダーと同じ次元空間で表現
    """
    
    def __init__(self, architecture='vit', embed_dim=512):
        super().__init__()
        
        self.architecture = architecture
        
        if architecture == 'vit':
            # Vision Transformer使用
            self.backbone = CustomVisionTransformer(
                img_size=224,
                patch_size=16, 
                embed_dim=embed_dim,
                num_heads=8,
                num_layers=12
            )
        elif architecture == 'resnet':
            # ResNet使用 (簡易版)
            self.backbone = CustomResNet(embed_dim=embed_dim)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
    def forward(self, images):
        """
        画像をエンコード
        
        Args:
            images: [batch_size, 3, 224, 224] - 正規化済み画像
        
        Returns:
            image_features: [batch_size, embed_dim] - 画像特徴量
        """
        
        return self.backbone(images)