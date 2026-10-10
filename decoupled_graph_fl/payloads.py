"""
payloads.py
═══════════════════════════════════════════════════════════════════
【SHARED MODULE】定义客户端↔服务端之间所有消息载荷的数据结构。
此模块同时被 Client 和 Server 使用，确保类型安全与载荷边界清晰。

设计原则：
  1. NodePayload（上行）: 绝不包含原始 token，只有语义嵌入张量
  2. GlobalPayload（下行）: 只包含聚合后的全局信息，不泄露其他
     Client 的本地分布
  3. 使用 dataclass + __post_init__ 校验保证载荷完整性
  4. 所有张量使用 bytes 序列化，通过 Ray Object Store 零拷贝传输
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


# ─────────────────────────────────────────────────────────────────
# 上行载荷：Client → Server
# ─────────────────────────────────────────────────────────────────
@dataclass
class NodePayload:
    """
    【CLIENT → SERVER 上行载荷】
    每个联邦客户端在每轮（Round）训练结束后向 TopologyCoordinator
    提交的数据包。

    ⚠️  安全边界：此结构中严禁包含 raw_tokens / input_ids / 任何
        可通过逆映射还原原始文本的中间表示。

    Attributes:
        client_id      : 客户端唯一标识符
        round_id       : 当前联邦训练轮次（防重放攻击用）
        embeddings     : 本地子图所有节点的语义嵌入，shape=[N, D]
                         N = 本地子图节点数，D = hidden_dim
        edge_index     : 本地子图的边索引（COO 稀疏格式），shape=[2, E]
                         E = 本地边数。仅包含拓扑结构，无节点特征！
        classifier_w   : 本地下游分类器的权重矩阵，shape=[D, C]
                         C = 分类类别数。用于 FedAvg 聚合。
        classifier_b   : 本地分类器偏置向量，shape=[C]
        domain_shift   : 域漂移量化指标（标量 float），
                         = ||δ_k||_2 / sqrt(D)，用于异常检测
        num_samples    : 本地训练样本数，用于加权 FedAvg
    """
    client_id: int
    round_id: int
    embeddings: torch.Tensor        # shape: [N, D]  (N 个节点，D 维嵌入)
    edge_index: torch.Tensor        # shape: [2, E]  (E 条边的 COO 格式)
    classifier_w: torch.Tensor      # shape: [D, C]  (分类器权重)
    classifier_b: torch.Tensor      # shape: [C]     (分类器偏置)
    domain_shift: float             # 域漂移量化指标 ∈ [0, 1)
    num_samples: int                # 本地样本数量（FedAvg 权重计算用）

    def __post_init__(self) -> None:
        """载荷合法性校验：确保张量维度一致性。"""
        assert self.embeddings.dim() == 2, (
            f"embeddings 应为 2D 张量 [N, D]，实际为 {self.embeddings.shape}"
        )
        assert self.edge_index.dim() == 2 and self.edge_index.shape[0] == 2, (
            f"edge_index 应为 [2, E] 格式，实际为 {self.edge_index.shape}"
        )
        assert self.classifier_w.dim() == 2, (
            f"classifier_w 应为 2D 张量 [D, C]，实际为 {self.classifier_w.shape}"
        )
        D = self.embeddings.shape[1]
        D_w = self.classifier_w.shape[0]
        assert D == D_w, (
            f"embeddings 维度 D={D} 应与 classifier_w 的行数 D_w={D_w} 匹配"
        )
        assert 0.0 <= self.domain_shift < 2.0, (
            f"domain_shift={self.domain_shift} 超出合理范围 [0, 2)"
        )
        assert self.num_samples > 0, "num_samples 必须大于 0"

    @property
    def num_nodes(self) -> int:
        """本地子图节点数量 N。"""
        return self.embeddings.shape[0]             # N

    @property
    def hidden_dim(self) -> int:
        """语义嵌入维度 D。"""
        return self.embeddings.shape[1]             # D

    @property
    def num_classes(self) -> int:
        """分类类别数 C。"""
        return self.classifier_w.shape[1]           # C


# ─────────────────────────────────────────────────────────────────
# 下行载荷：Server → Client
# ─────────────────────────────────────────────────────────────────
@dataclass
class GlobalPayload:
    """
    【SERVER → CLIENT 下行载荷】
    TopologyCoordinator 在每轮聚合后向各 Client 广播的全局信息。

    隐私设计：
      - global_embedding 是所有 Client 嵌入的加权平均，
        无法逆向追踪到任何单个 Client 的原始数据
      - anomaly_flags 只标记客户端 ID，不解释具体原因
        （防止信息泄露）
      - topology_scores 是服务端基于图结构计算的连通度分数，
        不依赖任何单个 Client 的私有特征

    Attributes:
        round_id          : 当前轮次编号
        global_embedding  : 全局聚合语义中心，shape=[D]
                            所有 Client 嵌入中心的加权 FedAvg 结果
        aggregated_w      : 聚合后的全局分类器权重，shape=[D, C]
        aggregated_b      : 聚合后的全局分类器偏置，shape=[C]
        topology_scores   : 各 Client 的拓扑连通置信分，Dict[client_id, float]
                            基于子图结构的重要性权重
        anomaly_flags     : 被 Byzantine 检测标记的异常 Client ID 列表
        accepted_clients  : 本轮参与聚合的有效 Client ID 列表
    """
    round_id: int
    global_embedding: torch.Tensor      # shape: [D]      (全局语义中心)
    aggregated_w: torch.Tensor          # shape: [D, C]   (全局分类器权重)
    aggregated_b: torch.Tensor          # shape: [C]      (全局分类器偏置)
    topology_scores: Dict[int, float]   # {client_id: score}
    anomaly_flags: List[int]            # 异常节点 ID 列表
    accepted_clients: List[int]         # 正常参与聚合的节点 ID 列表

    def __post_init__(self) -> None:
        """下行载荷校验：确保全局嵌入维度与分类器一致。"""
        assert self.global_embedding.dim() == 1, (
            f"global_embedding 应为 1D 向量 [D]，实际为 {self.global_embedding.shape}"
        )
        D = self.global_embedding.shape[0]
        D_w = self.aggregated_w.shape[0]
        assert D == D_w, (
            f"global_embedding 维度 D={D} 应与 aggregated_w 行数 D_w={D_w} 匹配"
        )
        assert self.aggregated_b.shape[0] == self.aggregated_w.shape[1], (
            "aggregated_b 维度应与 aggregated_w 列数（类别数 C）一致"
        )
