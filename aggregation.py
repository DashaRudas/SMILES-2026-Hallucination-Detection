"""
aggregation.py — Layer 13+16 pooling + geometric and statistic features.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _mean_pool_response(layer: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool response tokens for a single hidden layer."""
    h_resp = layer[response_mask]
    return h_resp.mean(dim=0) if h_resp.size(0) > 0 else torch.zeros(layer.size(1), device=layer.device)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int = 0,
) -> torch.Tensor:
    """Mean-pool response tokens from layers 13 and 16, then concatenate."""
    h_13 = hidden_states[13]  # (seq_len, hidden_dim)
    h_16 = hidden_states[16]  # (seq_len, hidden_dim)
    seq_len = attention_mask.size(0)
    
    response_mask = torch.zeros(seq_len, dtype=torch.bool, device=h_13.device)
    if prompt_len < seq_len:
        response_mask[prompt_len:seq_len] = attention_mask[prompt_len:].bool()

    pooled_13 = _mean_pool_response(h_13, response_mask)
    pooled_16 = _mean_pool_response(h_16, response_mask)
    return torch.cat([pooled_13, pooled_16], dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# Helper-функции для текстовых признаков
# ─────────────────────────────────────────────────────────────────────────────
def _compute_lexical_overlap(prompt: str, response: str) -> float:
    """
    Доля уникальных слов ответа, которые встречаются в промпте.
    Игнорирует стоп-слова (<4 букв) и пунктуацию.
    """
    if not prompt or not response:
        return 0.0
    
    # Простая токенизация: по пробелам, чистим пунктуацию
    clean = lambda w: w.lower().strip(".,!?;:\"'()[]{}-–—")
    p_words = {clean(w) for w in prompt.split() if len(clean(w)) > 3}
    r_words = {clean(w) for w in response.split() if len(clean(w)) > 3}
    
    if not r_words:
        return 0.0
    
    return len(p_words & r_words) / len(r_words)


def _compute_ttr(text: str) -> float:
    """
    Type-Token Ratio: доля уникальных слов в тексте.
    1.0 = все слова разные, 0.0 = все слова одинаковые.
    """
    if not text:
        return 0.0
    words = [w.lower().strip(".,!?;:\"'()[]{}-–—") for w in text.split()]
    words = [w for w in words if w]  # убрать пустые
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int = 0,
    prompt_text: str = "",
    response_text: str = "",
    tokenizer = None
) -> torch.Tensor:
    """
    Извлекает 6 геометрических признаков:
    1-2. norm, std ответа (13 слой, эмбеддинги)
    3.   cos_sim: контекст↔ответ (24 слой, эмбеддинги)
    4.   lexical_overlap (текст)
    5.   ttr_response (текст)
    6.   response_len_tokens (текст, нормированный)
    """
    seq_len = attention_mask.size(0)
    device = hidden_states.device

    # ── 1. Маски для промпта и ответа ───────────────────────────────────────
    prompt_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    prompt_mask[:min(prompt_len, seq_len)] = attention_mask[:min(prompt_len, seq_len)].bool()
    
    response_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    if prompt_len < seq_len:
        response_mask[prompt_len:seq_len] = attention_mask[prompt_len:seq_len].bool()

    # ── 2. Признаки с 13 слоя (норма и дисперсия ответа) ───────────────────
    h_13 = hidden_states[13]
    h_resp_13 = h_13[response_mask]

    if h_resp_13.size(0) == 0:
        return torch.zeros(6, device=device)
    
    pooled_13 = h_resp_13.mean(dim=0)
    norm_val = torch.norm(pooled_13, p=2)
    std_val = h_resp_13.std(dim=0, unbiased=False).mean()

    # ── 3. Косинусная схожесть контекст↔ответ (24 слой) ────────────────────
    h_24 = hidden_states[24]
    ctx_emb = h_24[prompt_mask].mean(dim=0) if prompt_mask.any() else torch.zeros(h_24.size(1), device=device)
    resp_emb = h_24[response_mask].mean(dim=0) if response_mask.any() else torch.zeros(h_24.size(1), device=device)
    
    if ctx_emb.norm() > 1e-6 and resp_emb.norm() > 1e-6:
        ctx_emb /= ctx_emb.norm()
        resp_emb /= resp_emb.norm()
        cos_sim = F.cosine_similarity(ctx_emb.unsqueeze(0), resp_emb.unsqueeze(0), dim=1).squeeze(0)
    else:
        cos_sim = torch.tensor(0.0, device=device)

    # ── 4. Текстовые признаки (вычисляются из сырых строк) ─────────────────
    # 4.1 Lexical overlap
    lexical_overlap = _compute_lexical_overlap(prompt_text, response_text)
    
    # 4.2 Type-Token Ratio ответа
    ttr_resp = _compute_ttr(response_text)
    
    # 4.3 Длина ответа в токенах (нормируем делением на 100 для масштаба ~[0, 3])
    resp_tokens = tokenizer(response_text, truncation=False, add_special_tokens=False)["input_ids"]
    response_len_norm = len(resp_tokens) / 100.0  # ~0.1-3.0 для типичных ответов

    # ── 5. Собираем все 6 признаков ────────────────────────────────────────
    return torch.tensor([
        norm_val,           # 0: норма эмбеддинга (13 слой)
        std_val,            # 1: дисперсия токенов (13 слой)
        cos_sim,            # 2: косинус контекст↔ответ (24 слой)
        lexical_overlap,    # 3: доля слов из промпта в ответе
        ttr_resp,           # 4: разнообразие словаря ответа
        response_len_norm,  # 5: длина ответа в токенах (норм.)
    ], device=device)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
    prompt_len: int = 0,
    prompt_text: str = "",
    response_text: str = "",
    tokenizer = None
) -> torch.Tensor:
    """Main entry point: aggregates hidden states + optional geometric features."""
    agg = aggregate(hidden_states, attention_mask, prompt_len)
    if use_geometric:
        geo = extract_geometric_features(
            hidden_states, attention_mask, prompt_len,
            prompt_text=prompt_text, response_text=response_text, tokenizer=tokenizer
        )
        return torch.cat([agg, geo], dim=0)  # [2 * 896 + 6] = 1798
    return agg  # [2 * 896] = 1792
