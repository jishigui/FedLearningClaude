"""
frozen_encoder.py
═══════════════════════════════════════════════════════════════════
【CLIENT-SIDE MODULE】本模块完全运行在各 Client 节点本地，输出的
语义嵌入向量才会被上传到 Server，原始 Token 永远不离开本地。

核心设计：Partitioned Vocabulary Blocks + Domain Shift Injection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
词表分块（Partitioned Vocabulary Blocks）原理：
  1. 将虚拟词表 V 均等划分为 K 个块，每块代表一个"域分区"
  2. 不同 Client 仅激活自己域对应的词表块，形成分块稀疏激活
  3. 在嵌入层注入可量化的域偏移向量 δ_k，使得：
       embed(token, domain=k) = base_embed(token) + δ_k
  4. 域漂移量化指标 domain_shift = ||δ_k||_2 / sqrt(hidden_dim)
     → 归一化后在 [0, 1) 范围内，可作为 Payload 的安全元数据
  5. 由于 δ_k 是随机初始化的，攻击者无法仅凭嵌入向量反推原始 token
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────
# 数据类：域分区配置
# ─────────────────────────────────────────────────────────────────
@dataclass
class VocabPartitionConfig:
    """描述一个词表分区（Vocabulary Partition Block）的配置。

    Attributes:
        vocab_size:   虚拟词表总大小（所有域共享同一索引空间）
        num_domains:  域的数量 = 词表分块数 K
        hidden_dim:   语义嵌入维度 D
        max_seq_len:  单次输入最大序列长度（用于模拟）
        domain_shift_scale: 域偏移向量初始化标准差，控制漂移强度
    """
    vocab_size: int = 512
    num_domains: int = 3          # K = 3 个域分块
    hidden_dim: int = 64          # 教学用小维度，生产中用 768/1024
    max_seq_len: int = 32
    domain_shift_scale: float = 0.5  # δ_k 初始化标准差


# ─────────────────────────────────────────────────────────────────
# 主模块：FrozenEncoder
# ─────────────────────────────────────────────────────────────────
class FrozenEncoder(nn.Module):
    """
    【CLIENT-SIDE】冻结编码器，将原始文本（此处以随机 token id 模拟）
    转化为语义嵌入向量。权重在联邦训练期间 **完全冻结（frozen）**，
    仅域偏移向量 δ 参与本地 fine-tune（可选）。

    为什么要 Frozen？
    ─────────────────
    1. 若 Encoder 权重可变并上传，攻击者可通过梯度逆向恢复原始 token
       （句子反演攻击 / Gradient Inversion Attack）
    2. 冻结 Encoder，只上传下游分类器权重，极大降低隐私泄露面

    词表分块域偏移机制：
    ─────────────────────
    token 索引空间 [0, V) 被划分为 K 块，每块大小 V//K：
      - Block 0: token [0,   V//K)   → 域 0（例如：科技领域）
      - Block 1: token [V//K, 2V//K) → 域 1（例如：医疗领域）
      - Block 2: token [2V//K, V)    → 域 2（例如：金融领域）

    每个 Client 在初始化时确定自己的 domain_id，并在前向传播中
    自动注入对应域的偏移向量 δ_{domain_id}，确保嵌入空间在域间不可互译。
    """

    def __init__(self, config: VocabPartitionConfig, domain_id: int) -> None:
        """
        Args:
            config:    词表分区配置
            domain_id: 当前 Client 所属域编号 ∈ [0, config.num_domains)
        """
        super().__init__()

        assert 0 <= domain_id < config.num_domains, (
            f"domain_id={domain_id} 越界，应在 [0, {config.num_domains}) 内"
        )

        self.config = config
        self.domain_id = domain_id
        self.block_size: int = config.vocab_size // config.num_domains

        # ── 基础嵌入层（共享词表空间，但域间激活区间不同）────────
        # shape: [vocab_size, hidden_dim]
        self.base_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_dim,
            padding_idx=0,
        )

        # ── 域偏移向量矩阵 Δ = [δ_0, δ_1, ..., δ_{K-1}] ─────────
        # shape: [num_domains, hidden_dim]
        # 每行 δ_k 是第 k 个域的偏移向量，由高斯分布初始化
        self.domain_shift_vectors = nn.Parameter(
            torch.randn(config.num_domains, config.hidden_dim)
            * config.domain_shift_scale,
            requires_grad=True,   # 允许本地域偏移微调，但不上传原值
        )

        # ── 轻量位置编码（正弦固定编码，不参与训练）────────────────
        # shape: [max_seq_len, hidden_dim]
        self.register_buffer(
            "pos_encoding",
            self._build_sinusoidal_pos_encoding(
                config.max_seq_len, config.hidden_dim
            ),
        )

        # ── 编码器主体：两层 LayerNorm + Linear 投影（完全冻结）────
        self.layer_norm1 = nn.LayerNorm(config.hidden_dim)
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim)

        # 冻结所有 base 权重（域偏移向量除外）
        self._freeze_base_weights()

    # ─────────────────────────────────────────────────────────────
    # 私有辅助：构建正弦位置编码
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _build_sinusoidal_pos_encoding(
        max_len: int, dim: int
    ) -> torch.Tensor:
        """构建固定正弦位置编码矩阵。

        PE[pos, 2i]   = sin(pos / 10000^(2i/dim))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/dim))

        Returns:
            Tensor of shape [max_len, dim]
        """
        pe = torch.zeros(max_len, dim)                  # [max_len, dim]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, dim, 2).float()
            * (-math.log(10000.0) / dim)
        )                                               # [dim//2]

        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维：sin
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维：cos
        return pe                                       # [max_len, dim]

    # ─────────────────────────────────────────────────────────────
    # 私有辅助：冻结 Base 权重
    # ─────────────────────────────────────────────────────────────
    def _freeze_base_weights(self) -> None:
        """将 base_embedding、proj、layer_norm 等权重设为不可训练。
        冻结后这些参数不会出现在 optimizer 的 param_groups 中。
        """
        frozen_modules = [self.base_embedding, self.proj,
                          self.layer_norm1, self.layer_norm2]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False

    # ─────────────────────────────────────────────────────────────
    # 核心前向传播
    # ─────────────────────────────────────────────────────────────
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        【CLIENT-SIDE 计算】将 token id 序列转化为语义嵌入向量。

        前向传播计算图：
          token_ids → base_embed → +pos_enc → layer_norm1
                   → proj → layer_norm2 → mean_pool → +δ_domain
                   → L2_normalize → output_embedding

        Args:
            token_ids: 模拟 token 序列，shape = [B, seq_len]
                       其中 B = batch size（即本地子图节点数）
                       seq_len ≤ config.max_seq_len

        Returns:
            embeddings: 归一化语义嵌入，shape = [B, hidden_dim]
                        ← 这是唯一会被上传到 Server 的张量！
        """
        B, seq_len = token_ids.shape                    # [B, seq_len]

        # ── Step 1: 基础词嵌入查表 ───────────────────────────────
        # 将 token id 限制在本域词表块范围内（域隔离关键操作）
        domain_offset: int = self.domain_id * self.block_size
        # 将任意 token id 通过取模+偏移映射到本域块内，模拟域词表激活
        domain_token_ids = (token_ids % self.block_size) + domain_offset
        # domain_token_ids shape: [B, seq_len]

        x = self.base_embedding(domain_token_ids)       # [B, seq_len, hidden_dim]

        # ── Step 2: 添加位置编码 ─────────────────────────────────
        x = x + self.pos_encoding[:seq_len, :].unsqueeze(0)
        # pos_encoding[:seq_len] shape: [seq_len, hidden_dim]
        # unsqueeze(0)           shape: [1, seq_len, hidden_dim]
        # x after add            shape: [B, seq_len, hidden_dim]

        # ── Step 3: LayerNorm → 线性投影 → LayerNorm ────────────
        x = self.layer_norm1(x)                         # [B, seq_len, hidden_dim]
        x = self.proj(x)                                # [B, seq_len, hidden_dim]
        x = self.layer_norm2(x)                         # [B, seq_len, hidden_dim]

        # ── Step 4: 平均池化 (Mean Pooling over seq_len) ─────────
        # 将序列压缩为单个节点向量，形成图节点特征
        x = x.mean(dim=1)                               # [B, hidden_dim]

        # ── Step 5: 注入域偏移向量 δ_{domain_id} ─────────────────
        # self.domain_shift_vectors[self.domain_id] shape: [hidden_dim]
        delta = self.domain_shift_vectors[self.domain_id]  # [hidden_dim]
        x = x + delta.unsqueeze(0)                      # [B, hidden_dim]
        # 域偏移使不同 Client 的嵌入分布产生可量化的系统性偏移，
        # 既标识了域身份，又防止了跨域嵌入空间的直接对齐攻击

        # ── Step 6: L2 归一化（统一超球面）──────────────────────
        x = F.normalize(x, p=2, dim=-1)                 # [B, hidden_dim]
        # 归一化后的嵌入向量位于 hidden_dim 维单位超球面上
        # 余弦相似度 = 内积，便于 Server 端的 Byzantine 检测

        return x                                        # [B, hidden_dim]

    # ─────────────────────────────────────────────────────────────
    # 域漂移量化指标
    # ─────────────────────────────────────────────────────────────
    def compute_domain_shift_metric(self) -> float:
        """
        计算本域的域漂移量化指标，用于 Payload 中的安全元数据。

        公式：domain_shift = ||δ_k||_2 / sqrt(hidden_dim)

        数学含义：
          - ||δ_k||_2 是域偏移向量的 L2 范数（域漂移绝对强度）
          - 除以 sqrt(hidden_dim) 进行维度归一化，使结果与维度无关
          - 结果范围 ≈ [0, domain_shift_scale] ≈ [0, 0.5]
          - Server 可用此指标检测异常域漂移（Byzantine 节点可能
            提交虚假的超大域漂移）

        Returns:
            float: 归一化域漂移指标
        """
        delta = self.domain_shift_vectors[self.domain_id]  # [hidden_dim]
        norm = torch.norm(delta, p=2)                   # scalar
        normalized = (norm / math.sqrt(self.config.hidden_dim)).item()
        return float(normalized)

    # ─────────────────────────────────────────────────────────────
    # 生成模拟 Token（用于教学验证，生产中替换为真实 tokenizer）
    # ─────────────────────────────────────────────────────────────
    def generate_mock_tokens(
        self,
        num_nodes: int,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        """
        生成随机模拟 token id，用于端到端测试。
        生产环境中此函数替换为真实 tokenizer 输出。

        Args:
            num_nodes: 本地子图节点数量 N
            seq_len:   序列长度，默认使用 config.max_seq_len

        Returns:
            token_ids: shape = [N, seq_len]
        """
        if seq_len is None:
            seq_len = self.config.max_seq_len
        # 在词表范围内均匀采样，模拟真实文本的 token 分布
        return torch.randint(
            low=1,                          # 0 是 padding_idx，跳过
            high=self.config.vocab_size,
            size=(num_nodes, seq_len),
        )                                   # [num_nodes, seq_len]
