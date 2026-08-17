import numpy as np
from pathlib import Path
import onnxruntime
from tokenizers import Tokenizer

class Embedder:
    def __init__(self, model_dir: str | Path = "models/bge-small-en-v1.5"):
        model_dir = Path(__file__).resolve().parent.parent / model_dir
        tokenizer_path = model_dir / "tokenizer.json"
        onnx_path = model_dir / "onnx" / "model.onnx"
        
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Ensure padding and truncation are enabled for batching
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=512)
        
        self.session = onnxruntime.InferenceSession(str(onnx_path))
        
    def _mean_pool_and_normalize(self, token_embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        # token_embeddings shape: [batch_size, seq_len, 384]
        # attention_mask shape: [batch_size, seq_len]
        
        # Expand attention mask to match embedding dimension
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        
        # Multiply embeddings by mask
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        
        # Sum of mask (to divide by)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        
        # Mean pooling
        mean_pooled = sum_embeddings / sum_mask
        
        # L2 normalize
        norms = np.linalg.norm(mean_pooled, axis=-1, keepdims=True)
        normalized = mean_pooled / np.clip(norms, a_min=1e-9, a_max=None)
        
        return normalized.astype(np.float32)

    def encode(self, text: str) -> np.ndarray:
        encoding = self.tokenizer.encode(text)
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)
        token_type_ids = np.array([encoding.type_ids], dtype=np.int64)
        
        outputs = self.session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        })
        token_embeddings = outputs[0]  # shape: [1, seq_len, 384]
        
        result = self._mean_pool_and_normalize(token_embeddings, attention_mask)
        return result[0]  # Return 1D array of shape (384,)

    def encode_batch(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = self.tokenizer.encode_batch(batch_texts)
            
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)
            
            outputs = self.session.run(None, {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids
            })
            token_embeddings = outputs[0]
            
            batch_embeddings = self._mean_pool_and_normalize(token_embeddings, attention_mask)
            all_embeddings.append(batch_embeddings)
            
        return np.vstack(all_embeddings)
