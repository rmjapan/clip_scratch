import torch
import regex as re
from typing import List, Dict, Union
import html


class CLIPTokenizer:
    """
    CLIP用のBPE (Byte Pair Encoding) トークナイザー
    
    処理の流れ:
    1. Text Cleaning: テキストの前処理・正規化
    2. BPE Encoding: サブワード単位への分割
    3. Token Mapping: 語彙辞書を使ってIDに変換
    4. Special Tokens: SOT/EOT/PADトークンの追加
    5. Sequence Padding: 固定長への調整
    """
    
    def __init__(self, vocab_file: str = None,bpe_merge_file: str = None, max_length: int = 77):
        """
        Args:
            vocab_file: BPE語彙ファイルのパス
            max_length: 最大シーケンス長 (CLIPでは77が標準)
        """
        self.max_length = max_length
        
        # デバッグ用フラグ
        self.verbose = False  # 全体デバッグ用フラグ
        self.clean_text_verbose = False  # テキスト前処理のデバッグ用フラグ
        self.bpe_verbose = False  # BPEエンコードのデバッグ用フラグ
        
        # Step 1: Special tokensを定義
        # CLIPで使用される特殊トークン
        self.pad_token = "<|pad|>"      # ID: 0
        self.sot_token = "<|startoftext|>"  # ID: 1  
        self.eot_token = "<|endoftext|>"    # ID: vocab_size-1
        self.unk_token = "<|unk|>"      # 未知語用
        
        # Step 2: BPE語彙辞書を読み込み
        # 
        # _file から token -> ID のマッピングを構築
        if vocab_file is None:
            self.vocab_file = "/home/ryuichi/face_editing/RelatedWork/clip_scratch/models/vocab/vocab.json"
        self.vocab = self._load_vocab(vocab_file)
        self.vocab_size = len(self.vocab)
        
        # Step 3: 逆引き辞書を作成 (ID -> token)
        # デコード時に使用（Key:ID⇒ID:Keyに変換して逆引き辞書を作成）
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        # Step 4: BPEマージルールを読み込み
        # サブワード分割のためのルール
        if bpe_merge_file is None:
            bpe_merge_file = "/home/ryuichi/face_editing/RelatedWork/clip_scratch/models/vocab/bpe_simple_vocab_16e6.txt"
        self.bpe_merges = self._load_bpe_merges(bpe_merge_file)
        
        # Step 5: テキスト前処理用の正規表現パターン（多言語対応）
        # regexモジュール使用でUnicode文字クラスに対応
        # 日本語: ひらがな・カタカナ・長音記号を一緒に処理
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{Hiragana}ー]+|[\p{Katakana}ー]+|[\p{Han}]+|\b\p{L}+\b| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
    def _load_vocab(self, vocab_file: str) -> Dict[str, int]:
        """
        CLIP語彙辞書（JSON）を読み込み
        
        Args:
            vocab_file: 語彙ファイルのパス（使用しない、固定パス）
            
        Returns:
            vocab: token -> ID のマッピング辞書
        """
        import json
        
        # CLIP公式の語彙辞書JSONファイルを使用
        vocab_json_path = "/home/ryuichi/face_editing/RelatedWork/clip_scratch/models/vocab/vocab.json"
        
        with open(vocab_json_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        print(f"Loaded vocabulary with {len(vocab)} tokens")
        return vocab

    def _load_bpe_merges(self, bpe_merge_file: str) -> Dict[tuple, int]:
        """
        BPEマージルールを読み込み
        
        Args:
            bpe_merge_file: BPEマージファイルのパス
            
        Returns:
            merges: {(token1, token2): priority} のマージペア辞書
        """
        if bpe_merge_file is None:
            bpe_merge_file = "/home/ryuichi/face_editing/RelatedWork/clip_scratch/models/vocab/bpe_simple_vocab_16e6.txt"
        if self.verbose:
            print(f"Loading BPE merges from: {bpe_merge_file}")
        
        merges = {}
        with open(bpe_merge_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # 1行目はヘッダーなのでスキップ
            for i, line in enumerate(lines[1:]):
                line = line.strip()#strip()  # 前後の空白を除去
                if line and not line.startswith('#'):
                    parts = line.split()#split()  # 空白で分割
                    if len(parts) == 2:
                        # 行番号を優先度として使用（小さいほど高優先度）
                        merges[(parts[0], parts[1])] = i
        
        print(f"Loaded {len(merges)} BPE merge rules")
        return merges
    
    def _clean_text(self, text: str) -> str:
        """
        テキストの前処理・正規化
        
        Args:
            text: 生のテキスト
            
        Returns:
            cleaned_text: 正規化されたテキスト
        """
        # Step 1: 基本的な正規化
        # 小文字化(lower)、余分な空白の除去（strip）
        if self.clean_text_verbose:
            print(f"{'before strip and lower':<25}:{text}")
        
        text = text.strip().lower()
        
        if self.clean_text_verbose:
            print(f"{'after strip and lower':<25}:{text}")
        
        # Step 2: 特殊文字の処理
        # HTMLエンティティのデコード、句読点の正規化
        if self.clean_text_verbose:
            print(f"{'before html unescape':<25}:{text}")
        
        text = html.unescape(text)
        
        if self.clean_text_verbose:
            print(f"{'after html unescape':<25}:{text}")
        
        # Step 3: Unicode正規化
        import unicodedata
        if self.clean_text_verbose:
            print(f"{'before unicode normalize':<25}:{text}")
        
        # Unicode正規化を行い、NFKC形式に変換
        text = unicodedata.normalize('NFKC', text)
        
        if self.clean_text_verbose:
            print(f"{'after unicode normalize':<25}:{text}")
        
        return text
  
    
    def _bpe_encode(self, text: str) -> List[str]:
        """
        BPE (Byte Pair Encoding) でサブワード分割
        
        Args:
            text: 前処理済みテキスト
            
        Returns:
            bpe_tokens: BPEトークンのリスト
        """
        # Step 1: 正規表現でトークンに分割(パターンにマッチする部分を抽出)
        # 単語、数字、句読点を適切に分離
        if self.bpe_verbose:
            print(f"{'before regex split':<25}:{text}")
        
        tokens = re.findall(self.pat, text)
        
        if self.bpe_verbose:
            print(f"{'after regex split':<25}:{tokens}")
        
        # Step 2: 各トークンをBPEでさらに分割
        bpe_tokens = []
        for token in tokens:
            # 文字レベルに分解(this ->[ 't', 'h', 'i', 's'])
            if self.bpe_verbose:
                print(f"{'token before BPE':<25}:{token}")
            
            word = list(token)
            
            if self.bpe_verbose:
                print(f"{'word before BPE':<25}:{word}")
          
            # BPEマージルールを適用
            while len(word) >= 2:
                # すべての隣接ペアを取得([t,h, i, s] -> [(t,h), (h,i), (i,s)]
                pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
                
                # if self.bpe_verbose:
                #     print(f"{'隣接ペア':<25}:{pairs}")
                
                # BPEマージルールに基づいて最も頻出するペアを取得
                  # (t,h)->0, (h,i)->1, (i,s)->2みたいな感じ→(t,h)
                bigram = min(pairs, key=lambda pair: self.bpe_merges.get(pair, float('inf')))
                
                if self.bpe_verbose:
                    print(f"{'最頻出ペア':<25}:{bigram}")
                
                #もしも最頻出ペアが語彙辞書に存在しない場合
                if bigram not in self.bpe_merges:
                    break        
                # マージを実行
                new_word = []
                i = 0
                while i < len(word):
                    #最終文字でない場合かつword[i]とword[i+1]が最頻出ペアならば
                    if i < len(word) - 1 and (word[i], word[i+1]) == bigram:
                        # 語どうしを結合
                        new_word.append(word[i] + word[i+1])
                        i += 2
                    #最終文字またはword[i]とword[i+1]が最頻出ペアでない場合
                    else:
                        # そのまま追加
                        new_word.append(word[i])
                        i += 1
                word = new_word
            bpe_tokens.extend(word)
   
        if self.bpe_verbose:
            print(f"{'final bpe_tokens':<25}:{bpe_tokens}")
        return bpe_tokens
      
    
    def encode(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        テキストをトークンIDに変換
        
        Args:
            text: 入力テキスト（文字列または文字列リスト）
            
        Returns:
            token_ids: トークンIDのテンソル [batch_size, max_length]
        """
        # 単一文字列の場合はリストに変換
        if isinstance(text, str):
            text = [text]
        
        batch_tokens = []
        for single_text in text:
            if self.verbose:
                print(f"{'looping text':<25}:{single_text}")
            # Step 1: テキスト前処理
            cleaned_text = self._clean_text(single_text) 
            if self.verbose:
                print(f"{'cleaned text':<25}:{cleaned_text}")
            # Step 2: BPEエンコーディング
            bpe_tokens = self._bpe_encode(cleaned_text)

            # Step 3: 語彙辞書でIDに変換
            token_ids = [self.vocab.get(self.sot_token, 1)]  # SOTトークンを先頭に
        
            for token in bpe_tokens:
                token_id = self.vocab.get(token, self.vocab.get(self.unk_token, 0))
                token_ids.append(token_id)
                if self.verbose:
                    print(f"{'token':<25}:{token}, id: {token_id}")
            # Step 4: EOTトークンを追加
            token_ids.append(self.vocab.get(self.eot_token, self.vocab_size - 1))
            if self.verbose:
                print(f"{'step3~4 token_ids':<25}:{token_ids}")
            # Step 5: パディング(ゼロ埋め)またはトランケート(はみ出た場合は切り捨て))
            if len(token_ids) > self.max_length:
                # 長すぎる場合はトランケート（EOTは保持）
                token_ids = token_ids[:self.max_length-1] + [token_ids[-1]]
            else:
                # 短い場合はパディング
                token_ids.extend([0] * (self.max_length - len(token_ids)))
            
            batch_tokens.append(token_ids)
        
        return torch.tensor(batch_tokens, dtype=torch.long)
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """
        トークンIDをテキストに変換
        
        Args:
            token_ids: トークンIDのテンソル [batch_size, seq_len]
            
        Returns:
            texts: デコードされたテキストのリスト
        """
        texts = []
        for sequence in token_ids:
            tokens = []
            # token id 列を処理して、idからtext に変換
            for token_id in sequence:
                # item()でスカラー値に変換
                token_id = token_id.item()     
                # 特殊IDの処理
                if token_id == 0:  # PADトークン or 未知語トークンの場合
                    continue
                # SOTトークンは無視（先頭にあるため）
                elif token_id == self.vocab.get(self.sot_token, 1):  # SOT
                    continue
                # EOTトークンは終了
                elif token_id == self.vocab.get(self.eot_token, self.vocab_size - 1):  # EOT
                    break
           
                else:
                    #idからトークンに変換
                    token = self.id_to_token.get(token_id, self.unk_token)
                    tokens.append(token)
            
            # BPEトークンを結合してテキストに復元
            if self.verbose:
                print(f"{'tokens before join':<25}:{tokens}")
            text = ''.join(tokens).replace('</w>', ' ').strip()
            if self.verbose:
                print(f"{'text after join':<25}:{text}")
            texts.append(text)
        
        return texts
        # pass
    
    def __len__(self) -> int:
        """語彙サイズを返す"""
        # return self.vocab_size
        pass
    def test_clip_tokenizer(self):
        """
        CLIPトークナイザーの基本的なテスト
        """
        # デバッグ用フラグを有効化
        self.verbose = True
        self.clean_text_verbose = True
        self.bpe_verbose = True
        # テスト用の簡単なテキスト
        test_text_list = ["roulston私の名前は宮内竜一です.笑顔がとても素敵でユーモアに溢れています。",
                          "１２３４56789は１０進数です、10進数です￥",
                          "This is a maple test sentence with some punctuation! 😊",
                          ]
        # トークン化
        token_ids = self.encode(test_text_list)
        
        # デコード
        decoded_text = self.decode(token_ids)
        
        print("Original Text:", test_text_list)
        print("Token IDs:", token_ids)
        print("Decoded Text:", decoded_text)
        
if __name__ == "__main__":
    tokenizer = CLIPTokenizer()
    tokenizer.test_clip_tokenizer()