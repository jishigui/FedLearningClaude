"""
federated_graph_node.py
═══════════════════════════════════════════════════════════════════
【CLIENT-SIDE RAY ACTOR】联邦图学习客户端节点

每个 FederatedGraphNode 是一个独立的 Ray Actor，代表一个联邦参与方。
它拥有：
  1. 本地子图结构（节点特征 + 边拓扑）
  2. FrozenEncoder（冻结语义编码器）
  3. 简单图卷积层（本地 GNN）
  4. 下游分类头（参与 FedAvg 聚合）

为什么用 Ray Actor 而不是 remote function？
─────────────────────────────────────────────
Ray Remote Function：无状态，每次调用独立，不保留中间状态。
Ray Actor：有状态，跨轮次保留模型权重、本地图结构、训练历史。
联邦学习需要跨多轮保持 Client 状态（本地模型、数据分布），
因此必须使用有状态的 Ray Actor。详见第三节自检问答。
"""

from __future__ import annotations

from typing import Optional, Tuple

import ray
import torch
import torch.nn as nn
import torch.nn.functional as F

from .frozen_encoder import FrozenEncoder, VocabPartitionConfig
from .payloads import GlobalPayload, NodePayload


# ─────────────────────────────────────────────────────────────────
# 辅助模块：简单图卷积层（SimpleGCNLayer）
# ─────────────────────────────────────────────────────────────────
class SimpleGCNLayer(nn.Module):
    """
    【CLIENT-SIDE】极简图卷积层，用于本地子图消息传递。

    GCN 公式（Kipf & Welling 2017 简化版）：
        H^{(l+1)} = σ( Â · H^{(l)} · W^{(l)} )
    其中：
        Â = D^{-1/2} · (A + I) · D^{-1/2}  (归一化邻接矩阵)
        H^{(l)} = 第 l 层节点特征矩阵，shape = [N, d_in]
        W^{(l)} = 可学习权重矩阵，shape = [d_in, d_out]
        σ = ReLU 激活函数

    在此实现中为教学简化版：
        - 用稀疏矩阵乘法替代完整 GCN 实现
        - 只做一次邻域聚合（1-hop aggregation）
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        # 图卷积权重矩阵 W，shape: [in_dim, out_dim]
        self.weight = nn.Parameter(
            torch.randn(in_dim, out_dim) * (1.0 / in_dim ** 0.5)
        )
        # 偏置向量，shape: [out_dim]
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(
        self,
        node_features: torch.Tensor,    # shape: [N, in_dim]
        adj_norm: torch.Tensor,         # shape: [N, N]  (归一化邻接矩阵 Â)
    ) -> torch.Tensor:
        """
        前向传播：H_out = σ( Â · H · W + b )

        Args:
            node_features: 节点特征矩阵，shape = [N, in_dim]
            adj_norm:      归一化邻接矩阵 Â，shape = [N, N]

        Returns:
            output: 更新后节点特征，shape = [N, out_dim]
        """
        # Step 1: 邻域聚合  AH = Â · H
        # [N, N] @ [N, in_dim] → [N, in_dim]
        agg = torch.mm(adj_norm, node_features)         # [N, in_dim]

        # Step 2: 线性变换  AHW = AH @ W
        # [N, in_dim] @ [in_dim, out_dim] → [N, out_dim]
        out = torch.mm(agg, self.weight) + self.bias    # [N, out_dim]

        # Step 3: ReLU 激活
        out = F.relu(out)                               # [N, out_dim]

        return out                                      # [N, out_dim]


# ─────────────────────────────────────────────────────────────────
# 辅助函数：构建归一化邻接矩阵
# ─────────────────────────────────────────────────────────────────
def build_normalized_adj(
    edge_index: torch.Tensor,           # shape: [2, E]
    num_nodes: int,
) -> torch.Tensor:
    """
    从 COO 格式边索引构建对称归一化邻接矩阵 Â。

    计算步骤：
      1. 构建邻接矩阵 A（加自环）：A_hat = A + I_N
      2. 计算度矩阵 D：D_{ii} = Σ_j A_hat_{ij}
      3. 归一化：Â = D^{-1/2} · A_hat · D^{-1/2}

    Args:
        edge_index: COO 格式边索引，shape = [2, E]
        num_nodes:  图中节点总数 N

    Returns:
        adj_norm: 归一化邻接矩阵，shape = [N, N]
    """
    N = num_nodes

    # Step 1: 构建稠密邻接矩阵（教学用，生产中用稀疏矩阵）
    A = torch.zeros(N, N)                               # [N, N]
    if edge_index.shape[1] > 0:
        # 填充双向边（无向图）
        row, col = edge_index[0], edge_index[1]         # [E], [E]
        A[row, col] = 1.0
        A[col, row] = 1.0                               # 对称化（无向图）

    # Step 2: 添加自环 Â = A + I
    A_hat = A + torch.eye(N)                            # [N, N]

    # Step 3: 计算度向量
    deg = A_hat.sum(dim=1)                              # [N]  每行求和 = 度数

    # Step 4: 计算 D^{-1/2}（度矩阵逆平方根）
    deg_inv_sqrt = deg.pow(-0.5)                        # [N]
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0    # 处理孤立节点（度=0）
    D_inv_sqrt = torch.diag(deg_inv_sqrt)               # [N, N]  对角矩阵

    # Step 5: 对称归一化 Â = D^{-1/2} · A_hat · D^{-1/2}
    # [N, N] @ [N, N] @ [N, N] → [N, N]
    adj_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt          # [N, N]

    return adj_norm                                     # [N, N]


# ─────────────────────────────────────────────────────────────────
# 主 Actor：FederatedGraphNode
# ─────────────────────────────────────────────────────────────────
@ray.remote
class FederatedGraphNode:
    """
    【CLIENT-SIDE RAY ACTOR】联邦图学习客户端节点。

    Actor 生命周期：
      - __init__: 初始化本地图结构、FrozenEncoder、GNN 和分类头
      - local_train: 在本地子图上执行前向传播 + 反向传播（仅更新分类器）
      - build_payload: 构造上行 NodePayload（不含原始 token）
      - receive_global: 接收 Server 的 GlobalPayload，更新本地分类器

    重要：FrozenEncoder 的权重在整个训练过程中保持冻结，
         只有 GCN 层和分类头的权重参与本地更新。
    """

    def __init__(
        self,
        client_id: int,
        domain_id: int,
        num_nodes: int,
        num_classes: int,
        vocab_config: VocabPartitionConfig,
        learning_rate: float = 1e-3,
    ) -> None:
        """
        初始化 FederatedGraphNode Actor。

        【CLIENT-SIDE 初始化，在 Ray Worker 进程中执行】

        Args:
            client_id:    当前 Client 的唯一 ID
            domain_id:    本 Client 所属的域编号（决定词表分区激活）
            num_nodes:    本地子图节点数量 N（模拟固定大小子图）
            num_classes:  分类任务的类别数 C
            vocab_config: FrozenEncoder 的词表分区配置
            learning_rate: 本地优化器学习率
        """
        self.client_id = client_id
        self.domain_id = domain_id
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.hidden_dim: int = vocab_config.hidden_dim
        self.round_id: int = 0                          # 当前轮次计数器

        # ── 组件 1: FrozenEncoder（冻结，不更新权重）──────────────
        # 【CLIENT-LOCAL】所有计算在本地，输出永不含原始 token
        self.encoder = FrozenEncoder(vocab_config, domain_id)

        # ── 组件 2: 本地子图 GCN 层（D → D 同维映射）─────────────
        # 【CLIENT-LOCAL】图结构学习，权重参与本地更新但不上传（仅上传分类器）
        self.gcn = SimpleGCNLayer(
            in_dim=self.hidden_dim,     # D
            out_dim=self.hidden_dim,    # D （保持维度一致）
        )

        # ── 组件 3: 下游分类头（线性层 D → C）────────────────────
        # 【CLIENT-LOCAL → 上传权重参与 FedAvg】
        self.classifier = nn.Linear(self.hidden_dim, num_classes, bias=True)
        # weight shape: [C, D]，注意 nn.Linear 的 weight 是转置存储的
        # 我们在 Payload 中将其转置为 [D, C] 以符合数学惯例

        # ── 组件 4: 本地优化器（仅更新 GCN + 分类器，不更新 Encoder）
        self.optimizer = torch.optim.Adam(
            list(self.gcn.parameters()) + list(self.classifier.parameters()),
            lr=learning_rate,
        )

        # ── 组件 5: 本地图结构（固定拓扑，模拟真实子图）─────────
        # 【CLIENT-LOCAL】生成随机连通图边索引（教学用）
        self.edge_index: torch.Tensor = self._generate_random_graph_edges(
            num_nodes, density=0.3
        )   # shape: [2, E]

        # ── 组件 6: 预构建归一化邻接矩阵（缓存复用）─────────────
        self.adj_norm: torch.Tensor = build_normalized_adj(
            self.edge_index, num_nodes
        )   # shape: [N, N]

        # ── 组件 7: 模拟本地标签（节点分类任务）─────────────────
        # 【CLIENT-LOCAL】真实场景中从本地数据集加载，此处随机生成
        self.local_labels: torch.Tensor = torch.randint(
            low=0, high=num_classes, size=(num_nodes,)
        )   # shape: [N]

        print(
            f"[Client {client_id}] 初始化完成 | 域={domain_id} | "
            f"节点数={num_nodes} | 边数={self.edge_index.shape[1]} | "
            f"类别数={num_classes} | 隐藏维度={self.hidden_dim}"
        )

    # ─────────────────────────────────────────────────────────────
    # 私有辅助：生成随机图拓扑
    # ─────────────────────────────────────────────────────────────
    def _generate_random_graph_edges(
        self, num_nodes: int, density: float = 0.3
    ) -> torch.Tensor:
        """
        生成随机无向图的边索引（Erdős-Rényi 模型）。

        每对节点 (i, j) 以概率 density 连接，保证图是连通的
        （通过添加一条 Hamiltonian 路径确保最小连通性）。

        Args:
            num_nodes: 节点数 N
            density:   边密度（连边概率）

        Returns:
            edge_index: COO 格式边索引，shape = [2, E]
        """
        edges: list[tuple[int, int]] = []

        # 先添加 Hamiltonian 路径确保连通性
        for i in range(num_nodes - 1):
            edges.append((i, i + 1))

        # 再按概率随机添加边
        for i in range(num_nodes):
            for j in range(i + 2, num_nodes):
                if torch.rand(1).item() < density:
                    edges.append((i, j))

        if not edges:
            # 至少保留一条边（防止空图）
            edges = [(0, 1)] if num_nodes > 1 else [(0, 0)]

        src = torch.tensor([e[0] for e in edges], dtype=torch.long)  # [E]
        dst = torch.tensor([e[1] for e in edges], dtype=torch.long)  # [E]
        return torch.stack([src, dst], dim=0)           # [2, E]

    # ─────────────────────────────────────────────────────────────
    # 核心方法 1: 本地训练
    # ─────────────────────────────────────────────────────────────
    def local_train(
        self, num_local_epochs: int = 3
    ) -> Tuple[float, float]:
        """
        【CLIENT-SIDE 计算】在本地子图上执行联邦学习的本地训练步骤。

        训练流程（全部在本地 Worker 进程中运行）：
          token_ids → FrozenEncoder → embeddings → GCN → classifier → loss

        数据流 Tensor Shape 追踪：
          token_ids         : [N, seq_len]
          embeddings        : [N, D]       ← Encoder 输出
          gcn_out           : [N, D]       ← GCN 聚合后
          logits            : [N, C]       ← 分类器输出
          loss              : scalar       ← 交叉熵损失

        Args:
            num_local_epochs: 本地训练 epoch 数（联邦学习通常 1~5）

        Returns:
            (final_loss, accuracy): 最后一个 epoch 的损失和准确率
        """
        self.gcn.train()
        self.classifier.train()
        # 注意：encoder 始终处于 eval 模式（参数已冻结）
        self.encoder.eval()

        final_loss: float = 0.0
        accuracy: float = 0.0

        for epoch in range(num_local_epochs):
            self.optimizer.zero_grad()

            # ── Step 1: 生成模拟 token（CLIENT-LOCAL，绝不上传）──
            token_ids = self.encoder.generate_mock_tokens(self.num_nodes)
            # token_ids shape: [N, seq_len]  ← 原始 token，永远不离开本地！

            # ── Step 2: FrozenEncoder 前向传播（CLIENT-LOCAL）────
            with torch.no_grad():
                # 注意：no_grad 范围内的 encoder 不参与梯度计算
                embeddings = self.encoder(token_ids)        # [N, D]
            # embeddings 是唯一可能被上传的张量，已经过域偏移 + L2 归一化

            # ── Step 3: GCN 图消息传递（CLIENT-LOCAL）────────────
            # adj_norm: [N, N]，embeddings: [N, D] → gcn_out: [N, D]
            gcn_out = self.gcn(embeddings, self.adj_norm)   # [N, D]

            # ── Step 4: 分类头前向传播（CLIENT-LOCAL）────────────
            # classifier.weight: [C, D]，输入: [N, D] → logits: [N, C]
            logits = self.classifier(gcn_out)               # [N, C]

            # ── Step 5: 计算交叉熵损失（CLIENT-LOCAL）────────────
            loss = F.cross_entropy(logits, self.local_labels)  # scalar

            # ── Step 6: 反向传播（只更新 GCN + Classifier）───────
            loss.backward()
            # 验证：encoder 参数梯度为 None（冻结确认）
            for param in self.encoder.base_embedding.parameters():
                assert param.grad is None, "FrozenEncoder 不应产生梯度！"

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(
                list(self.gcn.parameters()) + list(self.classifier.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            # ── 计算本 epoch 指标 ──────────────────────────────
            with torch.no_grad():
                preds = logits.argmax(dim=-1)               # [N]
                acc = (preds == self.local_labels).float().mean().item()

            final_loss = loss.item()
            accuracy = acc

            print(
                f"  [Client {self.client_id}] Epoch {epoch+1}/{num_local_epochs} | "
                f"loss={final_loss:.4f} | acc={accuracy:.3f}"
            )

        return final_loss, accuracy

    # ─────────────────────────────────────────────────────────────
    # 核心方法 2: 构造上行载荷
    # ─────────────────────────────────────────────────────────────
    def build_payload(self) -> NodePayload:
        """
        【CLIENT-SIDE】构造上行 NodePayload，准备发送给 Server。

        关键安全操作：
          1. 重新用 FrozenEncoder 生成当前节点嵌入（不依赖训练中间值）
          2. 仅提取分类器权重（不提取 GCN 权重，减少信息泄露面）
          3. 计算域漂移量化指标（Server 用于异常检测）
          4. 所有张量通过 .detach().clone() 切断计算图

        Returns:
            payload: NodePayload 实例（无原始 token）
        """
        self.encoder.eval()
        self.gcn.eval()

        with torch.no_grad():
            # 生成当前嵌入（CLIENT-LOCAL 计算，只有 result 上传）
            token_ids = self.encoder.generate_mock_tokens(self.num_nodes)
            # token_ids shape: [N, seq_len]  ← 此变量绝不出现在 payload 中！

            embeddings = self.encoder(token_ids)            # [N, D]

            # GCN 增强嵌入（使用本地图结构）
            gcn_embeddings = self.gcn(embeddings, self.adj_norm)  # [N, D]

            # 提取分类器权重
            # nn.Linear.weight shape: [C, D]，转置为 [D, C] 符合数学惯例
            classifier_w = self.classifier.weight.t().detach().clone()  # [D, C]
            classifier_b = self.classifier.bias.detach().clone()        # [C]

        # 计算域漂移量化指标（用于 Server 异常检测）
        domain_shift_metric: float = self.encoder.compute_domain_shift_metric()

        payload = NodePayload(
            client_id=self.client_id,
            round_id=self.round_id,
            embeddings=gcn_embeddings.detach().clone(),     # [N, D]  ← 上传
            edge_index=self.edge_index.clone(),             # [2, E]  ← 上传
            classifier_w=classifier_w,                     # [D, C]  ← 上传
            classifier_b=classifier_b,                     # [C]     ← 上传
            domain_shift=domain_shift_metric,              # float   ← 上传
            num_samples=self.num_nodes,                    # int     ← 上传
        )

        print(
            f"  [Client {self.client_id}] 构造 Payload | round={self.round_id} | "
            f"embeddings={tuple(payload.embeddings.shape)} | "
            f"domain_shift={domain_shift_metric:.4f}"
        )
        return payload

    # ─────────────────────────────────────────────────────────────
    # 核心方法 3: 接收全局更新
    # ─────────────────────────────────────────────────────────────
    def receive_global(self, global_payload: GlobalPayload) -> None:
        """
        【CLIENT-SIDE】接收 Server 的 GlobalPayload，更新本地分类器。

        更新策略：
          - 本地分类器权重 ← 全局聚合权重（完全替换，FedAvg 标准做法）
          - 本地 GCN 权重保持不变（保留本地图结构学习）
          - 若本 Client 被标记为 anomaly，打印警告但仍接收更新

        Args:
            global_payload: Server 广播的全局更新载荷
        """
        if self.client_id in global_payload.anomaly_flags:
            print(
                f"  ⚠️  [Client {self.client_id}] 被标记为异常节点！"
                f" 本轮更新仍执行，请检查本地数据分布。"
            )

        # 更新分类器权重（FedAvg 聚合结果）
        # global_payload.aggregated_w shape: [D, C]
        # nn.Linear.weight 存储为 [C, D]，需要转置
        with torch.no_grad():
            # [D, C].t() → [C, D]，赋值给 classifier.weight
            self.classifier.weight.copy_(
                global_payload.aggregated_w.t()    # [C, D]
            )
            self.classifier.bias.copy_(
                global_payload.aggregated_b        # [C]
            )

        # 轮次计数器递增
        self.round_id += 1

        topo_score = global_payload.topology_scores.get(self.client_id, 0.0)
        print(
            f"  [Client {self.client_id}] 接收全局更新 | round={self.round_id} | "
            f"拓扑置信分={topo_score:.3f}"
        )

    # ─────────────────────────────────────────────────────────────
    # 查询方法：获取客户端元信息（用于 Server 调度）
    # ─────────────────────────────────────────────────────────────
    def get_info(self) -> dict:
        """返回客户端基本元信息（不含任何敏感数据）。"""
        return {
            "client_id": self.client_id,
            "domain_id": self.domain_id,
            "num_nodes": self.num_nodes,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "round_id": self.round_id,
            "domain_shift": self.encoder.compute_domain_shift_metric(),
        }
