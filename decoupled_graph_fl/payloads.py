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

核心概念与维度速查：
  - 嵌入 (Embedding): 将高维离散数据（如文本、图节点）映射到低维连续的
    实数向量空间。语义或结构相近的节点，其嵌入向量在空间中的距离也相近。
    在联邦学习中，上传嵌入而非原始文本可有效实现隐私隔离。
  - N (num_nodes)   : 本地子图中的节点总数
  - E (num_edges)   : 本地子图中的有向边总数
  - D (hidden_dim)  : 每个节点的语义嵌入向量维度（特征维度）
  - C (num_classes) : 下游分类任务的类别总数
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
    提交的数据包，汇报本地训练成果。

    ⚠️  安全边界：此结构中严禁包含 raw_tokens / input_ids / 任何
        可通过逆映射还原原始文本的中间表示。

    Attributes:
        client_id      : 客户端唯一标识符（如 0, 1, 2...），用于区分数据来源。
        round_id       : 当前联邦训练轮次。用于防止旧轮次数据包延迟到达或被
                         恶意重放（防重放攻击）。
        embeddings     : 本地子图所有节点的语义嵌入矩阵，shape=[N, D]。
                         N = 本地子图节点数，D = hidden_dim（隐藏层特征维度）。
                         每一行代表一个节点的 D 维浓缩特征向量，由本地编码器
                         从原始数据提取而来，代替原始敏感文本上传以保护隐私。
        edge_index     : 本地子图的拓扑边索引（COO 稀疏矩阵格式），shape=[2, E]。
                         E = 本地边数。仅包含节点间的连线拓扑结构，无节点特征！
                         为什么是 [2, E] 而不是 [N, N] 邻接矩阵？
                           - 第 0 行：存放所有边的【起点 (Source)】节点编号
                           - 第 1 行：存放所有边的【终点 (Target)】节点编号
                           - 每一列（共 E 列）代表一条有向边（如第 k 列表示
                             edge_index[0, k] -> edge_index[1, k]）。
                           - 现实中的图大多是稀疏图，[2, E] 格式相比 [N, N] 稠密
                             邻接矩阵可省去海量 0 元素，极度节省内存与通信带宽。
        classifier_w   : 本地下游分类器的权重矩阵（Weight），shape=[D, C]。
                         C = 分类类别数。对应线性层 y = xW + b 中的 W，将 D 维
                         节点嵌入映射到 C 个类别得分（Logits），用于 FedAvg 聚合。
        classifier_b   : 本地分类器偏置向量（Bias），shape=[C]。
                         对应线性层 y = xW + b 中的偏置项 b。
        domain_shift   : 域漂移量化指标（标量 float，范围 [0, 2)），
                         计算公式 = ||δ_k||_2 / sqrt(D)。衡量当前客户端数据分布
                         偏离全局标准的程度，服务端用它辅助检测异常或投毒攻击。
        num_samples    : 本地有效训练样本数，用于加权 FedAvg（样本越多的客户端
                         在模型聚合时的权重通常越大）。
    """
    client_id: int
    round_id: int
    embeddings: torch.Tensor        # shape: [N, D]  (N 个节点，D 维语义嵌入)
    edge_index: torch.Tensor        # shape: [2, E]  (E 条边的 COO 稀疏格式：[起点行, 终点行])
    classifier_w: torch.Tensor      # shape: [D, C]  (分类器权重 W，将 D 维特征映射为 C 类)
    classifier_b: torch.Tensor      # 7shape: [C]     (分类器偏置 b)
    domain_shift: float             # 域漂移量化指标 ∈ [0, 2)
    num_samples: int                # 本地样本数量（FedAvg 权重计算用）

    def __post_init__(self) -> None:
        """
        载荷合法性校验：确保张量维度一致性。
        重点检查 embeddings 的特征维度 D 是否与分类器权重 classifier_w 的输入维度 D_w 匹配，
        以保证下游分类矩阵乘法 [N, D] @ [D, C] 能够正常进行。
        """
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
    TopologyCoordinator 在每轮收集所有 NodePayload、剔除异常节点并完成聚合后，
    向各 Client 广播的全局模型与状态信息，用于指导客户端下一轮本地训练。

    隐私设计：
      - global_embedding 是所有正常 Client 嵌入的加权平均，
        无法逆向追踪到任何单个 Client 的原始数据
      - anomaly_flags 只标记客户端 ID，不解释具体原因
        （防止信息泄露）
      - topology_scores 是服务端基于图结构计算的连通度分数d，
        不依赖任何单个 Client 的私有特征

    Attributes:
        round_id          : 当前轮次编号，标识这是哪一轮聚合出的全局模型。
        global_embedding  : 全局聚合语义中心（原型 Prototype），shape=[D]。
                            由所有正常 Client 的节点嵌入中心加权平均（FedAvg）得到。
                            客户端下一轮训练时会向该全局中心对齐，防止本地模型跑偏。
        aggregated_w      : 聚合后的全局分类器权重矩阵，shape=[D, C]。
                            由所有正常 Client 的 classifier_w 加权平均得到，客户端收到后
                            用它更新本地分类器。
        aggregated_b      : 聚合后的全局分类器偏置向量，shape=[C]。
                            由所有正常 Client 的 classifier_b 加权平均得到。
        topology_scores   : 各 Client 的拓扑连通置信分，Dict[client_id, float]。
                            服务端根据各客户端上传的图结构（edge_index）计算出的
                            子图结构重要性权重。
        anomaly_flags     : 异常客户端黑名单，List[int]。
                            本轮被拜占庭（Byzantine）防御机制判定为数据异常或恶意攻击的
                            Client ID 列表，其参数在本轮聚合中已被剔除。
        accepted_clients  : 正常客户端白名单，List[int]。
                            本轮通过安全检测、真正参与了全局模型聚合的有效 Client ID 列表。
    """
    round_id: int
    global_embedding: torch.Tensor      # shape: [D]      (全局语义中心向量)
    aggregated_w: torch.Tensor          # shape: [D, C]   (全局聚合分类器权重)
    aggregated_b: torch.Tensor          # shape: [C]      (全局聚合分类器偏置)
    topology_scores: Dict[int, float]   # {client_id: score} (各客户端拓扑连通度评分)
    anomaly_flags: List[int]            # 异常节点黑名单 ID 列表
    accepted_clients: List[int]         # 正常参与聚合的白名单节点 ID 列表

    def __post_init__(self) -> None:
        """下行载荷校验：确保全局嵌入维度 D 与全局分类器权重维度一致。"""
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
