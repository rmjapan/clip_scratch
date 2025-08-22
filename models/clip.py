import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import CLIPTokenizer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.attention_modules import CustomMultiHeadAttention


class CustomTransformerEncoderLayer(nn.Module):
    """
    学習用のTransformerEncoderLayer実装
    
    構成要素:
    1. Multi-Head Self-Attention: 入力シーケンス内の各要素間の関係を学習
    2. Position-wise Feed-Forward Network: 各位置で独立に適用される2層MLP
    3. Residual Connection: 各サブレイヤーの前後で残差接続
    4. Layer Normalization: 各サブレイヤー後に正規化
    
    処理の流れ:
    input → Self-Attention → Add & Norm → FFN → Add & Norm → output
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        # Step 1: Multi-Head Self-Attention層
        # 複数のattention headで異なる種類の関係性を並行学習(詳しくはAliciaYoutubeの動画参照)
        # 自作Multi-Head Self-Attention (学習用)
        self.self_attn = CustomMultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Step 2: Position-wise Feed-Forward Network
        # 各位置で独立に適用される2層MLP (embed_dim → ffn_dim → embed_dim)
        ffn_dim = embed_dim * 4  # CLIPでは通常4倍
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),  # CLIPではGELU活性化関数
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Step 3: Layer Normalization層
        # 各サブレイヤー後に適用する正規化
        self.norm1 = nn.LayerNorm(embed_dim)  # Self-Attention後
        self.norm2 = nn.LayerNorm(embed_dim)  # FFN後
        
        # Step 4: Dropout (regularization)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_key_padding_mask=None):
        """
        Forward pass: Transformer Encoder Layerの順伝播
        
        Args:
            x: 入力テンソル [batch_size, seq_len, embed_dim]
            src_key_padding_mask: attention mask [batch_size, seq_len]
                                 True positions are ignored in attention
        
        Returns:
            output: 変換された特徴量 [batch_size, seq_len, embed_dim]
        """
        
        # Step 1: Multi-Head Self-Attention + Residual Connection + Layer Norm
        # Query, Key, Valueは全て同じ入力xから生成 (Self-Attention)
        attn_output, _ = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=src_key_padding_mask#tokenidが0の位置をマスク(未知語あるいは、ゼロパディングトークンがこれに該当)
        )
        x = self.norm1(x + self.dropout(attn_output))  # Residual + LayerNorm（残差接続と学習効率化のための正規化）
        
        # Step 2: Position-wise FFN + Residual Connection + Layer Norm  
        # 各位置で独立にFFNを適用(2層mlp)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # Residual + LayerNorm
        
        return x



class TextEncoder(nn.Module):
    """
    Transformerベースのテキストエンコーダー
    
    処理の流れ:
    1. Token Embedding: 入力トークンを埋め込みベクトルに変換
    2. Positional Encoding: 位置情報を追加
    3. Transformer Layers: Multi-head attentionで文脈を学習
    4. EOTトークン位置で文章レベルの特徴量を抽出
    """
    
    def __init__(self, vocab_size=50000, embed_dim=512, num_heads=8, num_layers=12, max_seq_len=77):
        super().__init__()
        
        # Step 1: Token embedding layer
        # 各トークンIDを埋め込みベクトルに変換(NN.Embeddingはidとベクトルの対応を表すLookupテーブル、ベクトルは学習可能)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Step 2: Positional encoding (学習可能パラメータ)
        # 各位置の情報をベクトルで表現（位置埋め込みで、トークンの順序を考慮）
        self.positional_embedding = nn.Parameter(torch.randn(max_seq_len, embed_dim))
        
        # Step 3: Transformer encoder layers
        # Multi-head self-attentionで文脈関係を学習
        # 自作TransformerEncoderLayer (学習用)
        self.transformer = nn.ModuleList([
            CustomTransformerEncoderLayer(embed_dim, num_heads, dropout=0.1) 
            for _ in range(num_layers)
        ])
        
        # Step 4: Final layer normalization
        # 最終出力の正規化
        self.ln_final = nn.LayerNorm(embed_dim)
        
        # Mask for attention (padding tokens)
        self.max_seq_len = max_seq_len
    
    def forward(self, text_tokens):
        """
        テキストトークンから文章レベルの特徴量を抽出
        
        Args:
            text_tokens: [batch_size, seq_len] - トークン化されたテキスト
                       - 0: padding token
                       - 1: start of text token  
                       - 2~vocab_size-2: regular tokens
                       - vocab_size-1: end of text token
        
        Returns:
            text_features: [batch_size, embed_dim] - 文章レベルの特徴量
        """
        
        # Step 1: Token embeddingを取得
        # 各トークンIDを密なベクトル表現に変換
        x = self.token_embedding(text_tokens)  # [batch_size, seq_len, embed_dim]
        print(f"{'Token embeddings shape':<30}:{x.shape}")#IDをベクトルに変換
        
        # Step 2: Positional encodingを追加
        # 各位置の情報をembeddingに加算
        seq_len = x.shape[1]
        x = x + self.positional_embedding[:seq_len].unsqueeze(0)  # broadcast（[1, seq_len, embed_dim]になる)
        
        # Step 3: Attention maskを作成
        # padding token (0) の位置をマスク
        attention_mask = (text_tokens == 0)  # True for padding positions
        print(f"{'Attention mask':<30}:{attention_mask}")#Attention maskの形状
        
        # Step 4: Transformer encoderに通す
        # 各層を順番に適用
        for layer in self.transformer:
            x = layer(x, src_key_padding_mask=attention_mask)
        
        # Step 5: EOT (End of Text) トークンの位置で文章表現を抽出
        # CLIPでは最後の有効トークン位置の特徴量を使用
        print(f"{'text_tokens':<30}:{text_tokens}")#トークンの形状
        eot_token_pos = text_tokens.argmax(dim=-1)  # 各文のEOTトークン位置(Eotトークンは最大値IDであるため、argmaxで取得)
        print(f"{'EOT token positions':<30}:{eot_token_pos}")#
        text_features = x[torch.arange(x.shape[0]), eot_token_pos]
        
        # Step 6: Final layer normalization
        text_features = self.ln_final(text_features)
        
        return text_features
        # pass


class CLIP(nn.Module):
    """
    CLIP (Contrastive Language-Image Pre-training) モデル
    
    メイン構成要素:
    1. Text Encoder: テキストを埋め込みベクトルに変換
    2. Image Encoder: 画像を埋め込みベクトルに変換  
    3. Projection layers: 両方のエンコーダーの出力を同じ次元空間にマッピング
    4. Temperature parameter: contrastive lossの調整用
    """
    
    def __init__(self, vocab_size=50000, embed_dim=512, temperature=0.07, 
                 text_num_heads=8, text_num_layers=12, max_seq_len=77):
        super().__init__()
        #デバック用
        self.text_encode_vervose = True
        self.image_encode_vervose = True
        
        
        # Step 1: Text Encoderを初期化
        # Transformerベースのテキストエンコーダー
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim, 
            num_heads=text_num_heads,
            num_layers=text_num_layers,
            max_seq_len=max_seq_len
        )
        
        # Step 2: Image Encoderを初期化  
        # Vision Transformerまたは ResNetベースのエンコーダー
        # self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        
        # Step 3: Projection layersを初期化
        # テキストと画像の特徴量を共通の埋め込み空間にマッピング
        # Linear projection (bias=False for better performance)
        # self.text_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.image_projection = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Step 4: Temperature parameter (学習可能パラメータ)
        # Contrastive learningでの類似度スケーリング
        # log space で初期化して数値安定性を確保
        self.temperature = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))
        
    def encode_text(self, text):
        """
        テキストをエンコードして埋め込みベクトルを取得
        
        Args:
            text: 通常のテキスト
            
        Returns:
            text_features: 正規化された埋め込みベクトル [batch_size, embed_dim]
        """
        #Step 0: トークン化処理
        text = CLIPTokenizer().encode(text)# トークン化処理
        print(f"{'tokenized textの形状データ':<30}:{text.shape}")
        # Step 1: テキストエンコーダーでテキスト特徴量を抽出
        # Transformerを通して文脈を考慮した文章表現を取得
        text_features = self.text_encoder(text)  # [batch_size, embed_dim]
        
        # Step 2: プロジェクション層で最終埋め込み次元にマッピング
        # テキストエンコーダーの出力をCLIP空間に投影
        # text_features = self.text_projection(text_features)  # [batch_size, embed_dim]
        
        # Step 3: L2正規化でベクトルを単位球面上に配置
        # コサイン類似度計算のため、全てのベクトルのノルムを1に統一
        # text_features = F.normalize(text_features, dim=-1)  # [batch_size, embed_dim]
        
        # return text_features
        pass
        
    def encode_image(self, image):
        """
        画像をエンコードして埋め込みベクトルを取得
        
        Args:
            image: 前処理済み画像 [batch_size, channels, height, width]
                  - 通常は [batch_size, 3, 224, 224] (RGB画像)
                  - 正規化済み (ImageNet統計値で標準化)
            
        Returns:
            image_features: 正規化された埋め込みベクトル [batch_size, embed_dim]
        """
        # Step 1: 画像エンコーダーで画像特徴量を抽出
        # Vision TransformerまたはResNetで視覚的特徴を抽出
        # image_features = self.image_encoder(image)  # [batch_size, embed_dim]
        
        # Step 2: プロジェクション層で最終埋め込み次元にマッピング
        # 画像エンコーダーの出力をCLIP空間に投影
        # image_features = self.image_projection(image_features)  # [batch_size, embed_dim]
        
        # Step 3: L2正規化でベクトルを単位球面上に配置
        # テキスト特徴量と同じ空間での比較のため
        # image_features = F.normalize(image_features, dim=-1)  # [batch_size, embed_dim]
        
        # return image_features
        pass
        
    def forward(self, image, text):
        """
        Forward pass: 画像とテキストの類似度を計算
        
        Args:
            image: バッチ画像 [batch_size, channels, height, width]
            text: バッチテキスト [batch_size, seq_len]
            
        Returns:
            logits_per_image: 画像から見たテキストとの類似度 [batch_size, batch_size]
            logits_per_text: テキストから見た画像との類似度 [batch_size, batch_size]
        """
        # Step 1: 画像とテキストをそれぞれエンコード
        # 両方とも正規化された埋め込みベクトルを取得
        # image_features = self.encode_image(image)  # [batch_size, embed_dim]
        # text_features = self.encode_text(text)     # [batch_size, embed_dim]
        
        # Step 2: 類似度行列を計算 (正規化済みなのでコサイン類似度)
        # 各画像と各テキストの全ペアの類似度を計算
        # similarity_matrix = torch.matmul(image_features, text_features.T)  # [batch_size, batch_size]
        
        # Step 3: Temperature scalingを適用
        # 学習可能なtemperatureパラメータで類似度をスケーリング
        # 高いtemperatureで確率分布をシャープに、低いとソフトに
        # logits_per_image = similarity_matrix * self.temperature.exp()
        # logits_per_text = logits_per_image.T  # 転置で相互の類似度行列
        
        # return logits_per_image, logits_per_text
        pass


def contrastive_loss(logits_per_image, logits_per_text):
    """
    Contrastive Loss (InfoNCE) を計算
    
    CLIPの核心となる損失関数:
    - 正解ペア(diagonal)の類似度を最大化
    - 不正解ペア(off-diagonal)の類似度を最小化
    
    InfoNCE原理:
    - バッチ内で対応する画像-テキストペアのみが正例
    - 他の全てのペアは負例として扱う
    - Softmax cross entropyで正例の確率を最大化
    
    Args:
        logits_per_image: [batch_size, batch_size] - 画像から見たテキスト類似度行列
        logits_per_text: [batch_size, batch_size] - テキストから見た画像類似度行列
        
    Returns:
        loss: スカラー値の損失
    """
    # Step 1: 正解ラベルを作成 (対角線が正解ペア)
    # バッチ内のi番目の画像はi番目のテキストと対応
    # labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    
    # Step 2: 画像→テキスト方向の cross entropy loss
    # 各画像について、正しいテキストの確率を最大化
    # logits_per_image[i]がi番目の画像から見た全テキストとの類似度
    # loss_i = F.cross_entropy(logits_per_image, labels)
    
    # Step 3: テキスト→画像方向の cross entropy loss
    # 各テキストについて、正しい画像の確率を最大化  
    # logits_per_text[j]がj番目のテキストから見た全画像との類似度
    # loss_t = F.cross_entropy(logits_per_text, labels)
    
    # Step 4: 両方向の損失の平均 (対称的な学習)
    # 画像→テキストとテキスト→画像の両方向を同等に重視
    # loss = (loss_i + loss_t) / 2
    
    # return loss
    pass

if __name__ == "__main__":
    text=["a photo of a cat", "a photo of a dog with a ball"]
    model = CLIP()
    model.encode_text(text)