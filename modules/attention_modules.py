import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomMultiHeadAttention(nn.Module):
    """
    学習用のMulti-Head Self-Attention実装
    
    処理の流れ:
    1. Linear変換でQuery, Key, Valueを生成
    2. embed_dimをnum_headsに分割 
    3. 各ヘッドで並行してScaled Dot-Product Attentionを計算
    4. 各ヘッドの出力を連結
    5. 最終的な線形変換で出力
    
    数学的定義:
    Attention(Q,K,V) = softmax(QK^T / √d_k)V
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        # パラメータ設定
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 各ヘッドの次元
        # assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Step 1: Query, Key, Value変換用の線形層
        # 各ヘッド用のQ,K,Vを一括で生成するための線形変換
        self.q_linear = nn.Linear(embed_dim, embed_dim, bias=False)  # Query変換
        self.k_linear = nn.Linear(embed_dim, embed_dim, bias=False)  # Key変換  
        self.v_linear = nn.Linear(embed_dim, embed_dim, bias=False)  # Value変換
        
        # Step 2: 最終出力用の線形層
        # 連結されたマルチヘッド出力を最終次元に変換
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=False)
        
        # Step 3: Dropout (正則化)
        self.dropout = nn.Dropout(dropout)
        
        # Step 4: スケーリング係数
        # Attention計算時の分散を抑制するためのスケーリング
        self.scale = (self.head_dim) ** -0.5  # 1/√d_k
        
    def forward(self, query, key, value, key_padding_mask=None):
        """
        Multi-Head Self-Attentionの順伝播
        
        Args:
            query: [batch_size, seq_len, embed_dim] - クエリテンソル
            key:   [batch_size, seq_len, embed_dim] - キーテンソル  
            value: [batch_size, seq_len, embed_dim] - バリューテンソル
            key_padding_mask: [batch_size, seq_len] - パディング位置のマスク
            
        Returns:
            output: [batch_size, seq_len, embed_dim] - 変換された特徴量
            attention_weights: [batch_size, num_heads, seq_len, seq_len] - 注意重み
        """
        
        batch_size, seq_len, embed_dim = query.shape
        
        # Step 1: Query, Key, Valueの線形変換
        # 入力をQ, K, Vに変換（入力を直接使用せず、線形変換を適用）
        Q = self.q_linear(query)  # [batch, seq_len, embed_dim]
        K = self.k_linear(key)    # [batch, seq_len, embed_dim] 
        V = self.v_linear(value)  # [batch, seq_len, embed_dim]
        
        # Step 2: マルチヘッド用に次元を分割・再配置
        # embed_dim → num_heads × head_dim に分割
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # Step 3: Scaled Dot-Product Attention
        # 各ヘッドで並行してattentionを計算（batch_size, num_heads, seq_len, head_dim）
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, key_padding_mask)
        
        # Step 4: マルチヘッド出力の連結
        # 各ヘッドの出力を結合して元の次元に戻す
        attention_output = attention_output.transpose(1, 2)  # [batch, seq_len, num_heads, head_dim]
        attention_output = attention_output.contiguous().view(batch_size, seq_len, self.embed_dim)  # [batch, seq_len, embed_dim]
        
        # Step 5: 最終線形変換
        # 連結された出力を最終的な表現に変換
        output = self.out_linear(attention_output)  # [batch, seq_len, embed_dim]
        
        return output, attention_weights
    def scaled_dot_product_attention(self, Q, K, V, key_padding_mask=None):
        """
        Scaled Dot-Product Attentionの計算
        
        数学式: Attention(Q,K,V) = softmax(QK^T / √d_k)V
        
        Args:
            Q: [batch, num_heads, seq_len, head_dim] - Query
            K: [batch, num_heads, seq_len, head_dim] - Key
            V: [batch, num_heads, seq_len, head_dim] - Value
            key_padding_mask: [batch, seq_len] - パディングマスク
            
        Returns:
            output: [batch, num_heads, seq_len, head_dim] - attention適用後の値
            attention_weights: [batch, num_heads, seq_len, seq_len] - attention重み
            
        簡単に説明すると、Queryは「ヒーローが活躍する本」というリクエストで、Keyは本のタイトル、Valueはその内容。
        最も近いタイトル（Key）を見つけて、その内容（Value）を返すのがAttentionの役割。
        Ｑ:横ベクトル:[batch,8,77,512]
        Ｋ:横べクトル:[batch,8,77,512]、
        Ｖ:横ベクトル:[batch,8,77,512]
        Q1：[batch,1,77,512]、K1：[batch,1,77,512]、V1：[batch,1,77,512]
        
        
        
        """
        
        # Step 1: Query-Key類似度の計算
        # Q と K の内積でattention scoreを計算（transposeでseqlenとhead_dimを入れ替える）
        scores = torch.matmul(Q, K.transpose(-2, -1))  # [batch, num_heads, seq_len, seq_len]
        
        # Step 2: スケーリング
        # √d_k で割って分散を調整（勾配安定化）
        scores = scores * self.scale
        
        # Step 3: パディングマスクの適用
        # パディング位置は-infにして、softmax後に0になるようにする
        if key_padding_mask is not None:
            # key_padding_mask: [batch, seq_len] → [batch, 1, 1, seq_len]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores.masked_fill_(mask, float('-inf'))
        
        # Step 4: Softmax で確率分布に変換
        # 各クエリについて、どのキーに注目するかの確率分布（dim=-1はKey方向）
        attention_weights = F.softmax(scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]
        
        # Step 5: Dropout適用 (学習時の正則化)
        attention_weights = self.dropout(attention_weights)
        
        # Step 6: Value と attention重みの加重和
        # 注目する位置に基づいてValueを統合
        output = torch.matmul(attention_weights, V)  # [batch, num_heads, seq_len, head_dim]
        
        return output, attention_weights

    