"""
topology_coordinator.py
═══════════════════════════════════════════════════════════════════
【SERVER-SIDE RAY ACTOR】拓扑协调器 / 联邦聚合服务器

TopologyCoordinator 是整个框架的神经中枢，负责：
  1. 接收各 Client 上传的 NodePayload（语义嵌入，绝无原始 token）
  2. Byzantine 检测：基于余弦相似度 + 域漂移过滤异常节点
  3. 拓扑感知图聚合：构建跨 Client 全局图，计算拓扑置信分
  4. FedAvg 加权聚合：生成全局分类器权重
  5. 广播 GlobalPayload 给所有 Client

安全威胁模型（此实现所防御的攻击）：
  ┌─────────────────────────────────────────────────────────┐
  │ 攻击类型              防御机制                          │
  │ ──────────────        ─────────────────────────────     │
  │ 句子反演攻击           FrozenEncoder + 域偏移            │
  │ 拜占庭攻击             余弦相似度过滤（阈值 θ）          │
  │ 数据投毒攻击           域漂移量化异常检测                │
  │ 梯度窃取攻击           仅传递嵌入，不传梯度              │
  │ 重放攻击              round_id 验证                     │
  └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import ray
import torch
import torch.nn.functional as F

from .payloads import GlobalPayload, NodePayload


# ─────────────────────────────────────────────────────────────────
# 主 Actor：TopologyCoordinator
# ─────────────────────────────────────────────────────────────────
@ray.remote
class TopologyCoordinator:
    """
    【SERVER-SIDE RAY ACTOR】拓扑协调器，负责全局聚合与拓扑安全防御。

    为什么 Server 也要用 Ray Actor？
    ───────────────────────────────
    Server 需要跨轮次维护状态：
      - 历史聚合结果（用于检测客户端突变）
      - 各 Client 的历史置信分（滑动平均）
      - round_id 计数器（防重放）
    因此 Server 同样需要有状态的 Ray Actor，而非无状态 remote function。

    Actor 内部状态（SERVER-LOCAL，客户端不可见）：
      - round_id: 全局轮次计数
      - client_trust_scores: 各 Client 历史置信分（异常检测用）
      - last_global_w: 上一轮全局分类器权重（用于检测突变）
    """

    def __init__(
        self,
        num_clients: int,
        hidden_dim: int,
        num_classes: int,
        byzantine_threshold: float = 0.3,
        domain_shift_max: float = 1.5,
    ) -> None:
        """
        初始化 TopologyCoordinator。

        【SERVER-SIDE 初始化，在 Ray Worker 进程中执行】

        Args:
            num_clients:       联邦参与 Client 总数
            hidden_dim:        语义嵌入维度 D
            num_classes:       分类类别数 C
            byzantine_threshold: 余弦相似度过滤阈值 θ ∈ [0, 1]
                                 低于此阈值的嵌入被判定为异常
                                 （攻击者的恶意嵌入方向偏离中心）
            domain_shift_max:  域漂移上限，超过此值触发异常警报
        """
        self.num_clients = num_clients
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.byzantine_threshold = byzantine_threshold
        self.domain_shift_max = domain_shift_max

        # ── Server 内部状态（CLIENT 不可见）──────────────────────
        self.round_id: int = 0

        # 各 Client 的历史置信分，初始化为 1.0（完全信任）
        # shape: dict {client_id: float}
        self.client_trust_scores: Dict[int, float] = {
            i: 1.0 for i in range(num_clients)
        }

        # 上一轮全局分类器权重（用于检测分类器权重的突变）
        # shape: [D, C]
        self.last_global_w: Optional[torch.Tensor] = None

        # 历史全局嵌入中心（用于 Byzantine 基准计算）
        # shape: [D]
        self.global_embedding_center: Optional[torch.Tensor] = None

        print(
            f"[Server] TopologyCoordinator 初始化 | "
            f"客户端数={num_clients} | D={hidden_dim} | C={num_classes} | "
            f"Byzantine阈值={byzantine_threshold}"
        )

    # ─────────────────────────────────────────────────────────────
    # 核心方法 1: Byzantine 检测
    # ─────────────────────────────────────────────────────────────
    def _detect_byzantine_nodes(
        self,
        payloads: List[NodePayload],
    ) -> Tuple[List[int], List[int]]:
        """
        【SERVER-SIDE 计算】基于余弦相似度 + 域漂移的 Byzantine 节点检测。

        检测算法：
          1. 计算各 Client 嵌入的全局中心向量（未过滤的粗略中心）
          2. 计算每个 Client 嵌入中心与全局中心的余弦相似度
          3. 相似度低于阈值 θ → 判定为方向性异常（可能的投毒攻击）
          4. 域漂移指标超过 domain_shift_max → 判定为域漂移异常
          5. 两种异常的并集 = 被标记的 Byzantine 节点

        数学分析：
          - 余弦相似度 cos(θ) = (u·v) / (||u|| · ||v||)
          - 对于超球面上的向量（FrozenEncoder 已 L2 归一化），
            cos(θ) = u · v（内积即余弦相似度）
          - Byzantine 攻击者的嵌入向量方向会显著偏离良性客户端的中心

        Args:
            payloads: 所有客户端上传的 NodePayload 列表

        Returns:
            (accepted_ids, anomaly_ids): 正常客户端 ID 列表，异常客户端 ID 列表
        """
        print(f"\n  [Server] 执行 Byzantine 检测 | 候选客户端数={len(payloads)}")

        # ── Step 1: 计算各 Client 的嵌入中心向量 ─────────────────
        # client_centers[i] shape: [D]（第 i 个 client 的嵌入均值）
        client_centers: Dict[int, torch.Tensor] = {}
        for payload in payloads:
            # payload.embeddings shape: [N_i, D]
            # mean(dim=0) → [D]（节点嵌入的均值，代表该 Client 的语义中心）
            center = payload.embeddings.mean(dim=0)         # [D]
            center = F.normalize(center, p=2, dim=0)        # [D]（L2 归一化）
            client_centers[payload.client_id] = center

        # ── Step 2: 计算全局嵌入中心（简单平均，稳健估计）────────
        all_centers = torch.stack(list(client_centers.values()), dim=0)
        # all_centers shape: [K, D]  K = 当前轮参与 client 数

        global_center = all_centers.mean(dim=0)             # [D]
        global_center = F.normalize(global_center, p=2, dim=0)  # [D]

        # 更新历史全局中心（滑动平均，α = 0.7）
        if self.global_embedding_center is None:
            self.global_embedding_center = global_center
        else:
            alpha = 0.7
            self.global_embedding_center = (
                alpha * self.global_embedding_center
                + (1 - alpha) * global_center
            )   # [D]

        # ── Step 3: 计算余弦相似度并过滤 ────────────────────────
        anomaly_ids: List[int] = []
        accepted_ids: List[int] = []

        for payload in payloads:
            client_id = payload.client_id
            center = client_centers[client_id]              # [D]

            # 与全局中心的余弦相似度（两者均 L2 归一化，故内积 = 余弦相似度）
            # [D] · [D] → scalar
            cosine_sim: float = torch.dot(
                center, self.global_embedding_center
            ).item()

            # 域漂移检测
            domain_shift: float = payload.domain_shift
            is_direction_anomaly: bool = cosine_sim < self.byzantine_threshold
            is_shift_anomaly: bool = domain_shift > self.domain_shift_max

            # round_id 检测（防重放攻击）
            is_replay: bool = payload.round_id != self.round_id

            if is_direction_anomaly or is_shift_anomaly or is_replay:
                anomaly_ids.append(client_id)
                # 降低信任分（惩罚机制，指数衰减）
                self.client_trust_scores[client_id] *= 0.5
                reason = []
                if is_direction_anomaly:
                    reason.append(f"余弦相似度={cosine_sim:.3f}<{self.byzantine_threshold}")
                if is_shift_anomaly:
                    reason.append(f"域漂移={domain_shift:.3f}>{self.domain_shift_max}")
                if is_replay:
                    reason.append(f"重放攻击(round={payload.round_id}!={self.round_id})")
                print(
                    f"    ❌ [Server] Client {client_id} 被标记为异常 | "
                    f"原因: {', '.join(reason)}"
                )
            else:
                accepted_ids.append(client_id)
                # 恢复信任分（上界为 1.0）
                self.client_trust_scores[client_id] = min(
                    1.0, self.client_trust_scores[client_id] * 1.1
                )
                print(
                    f"    ✅ [Server] Client {client_id} 验证通过 | "
                    f"余弦相似度={cosine_sim:.3f} | 域漂移={domain_shift:.4f}"
                )

        return accepted_ids, anomaly_ids

    # ─────────────────────────────────────────────────────────────
    # 核心方法 2: 拓扑置信分计算
    # ─────────────────────────────────────────────────────────────
    def _compute_topology_scores(
        self,
        payloads: List[NodePayload],
        accepted_ids: List[int],
    ) -> Dict[int, float]:
        """
        【SERVER-SIDE 计算】基于局部图拓扑结构计算各 Client 的置信分。

        拓扑置信分的计算思路：
          1. 对于每个 Client，统计其本地子图的平均度数（Average Degree）
             → 高连通度的子图包含更丰富的结构信息，权重更高
          2. 结合历史 Byzantine 信任分
          3. 最终置信分 = 归一化拓扑分 × 历史信任分

        拓扑分计算：
          平均度数 = (2 × 边数 E) / 节点数 N  （无向图，每条边贡献度数 2）
          → 归一化为 [0, 1] 范围

        Args:
            payloads:     所有上传的 NodePayload 列表
            accepted_ids: 通过 Byzantine 检测的 Client ID 列表

        Returns:
            topology_scores: Dict[client_id, float]，各 Client 置信分
        """
        payload_map: Dict[int, NodePayload] = {
            p.client_id: p for p in payloads
        }
        topology_scores: Dict[int, float] = {}

        # 计算所有 accepted client 的最大平均度数（用于归一化）
        max_avg_degree: float = 1.0
        avg_degrees: Dict[int, float] = {}

        for client_id in accepted_ids:
            payload = payload_map[client_id]
            N = payload.num_nodes
            E = payload.edge_index.shape[1]         # 边数
            avg_degree = (2.0 * E) / max(N, 1)     # 平均度数（无向图）
            avg_degrees[client_id] = avg_degree
            max_avg_degree = max(max_avg_degree, avg_degree)

        for client_id in accepted_ids:
            avg_degree = avg_degrees.get(client_id, 0.0)
            # 归一化拓扑分 ∈ [0, 1]
            topo_score = avg_degree / max_avg_degree
            # 乘以历史信任分
            trust = self.client_trust_scores.get(client_id, 1.0)
            topology_scores[client_id] = topo_score * trust

        # 被拒绝的 Client 置信分为 0
        for payload in payloads:
            if payload.client_id not in topology_scores:
                topology_scores[payload.client_id] = 0.0

        return topology_scores

    # ─────────────────────────────────────────────────────────────
    # 核心方法 3: 拓扑感知 FedAvg 聚合
    # ─────────────────────────────────────────────────────────────
    def _topology_aware_fedavg(
        self,
        payloads: List[NodePayload],
        accepted_ids: List[int],
        topology_scores: Dict[int, float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        【SERVER-SIDE 计算】拓扑感知的加权 FedAvg 聚合。

        标准 FedAvg（McMahan et al., 2017）:
          w_global = Σ_k ( n_k / Σ n_k ) × w_k
          其中 n_k 是 Client k 的本地样本数

        拓扑感知改进：
          w_global = Σ_k ( n_k × s_k / Σ (n_k × s_k) ) × w_k
          其中 s_k 是 Client k 的拓扑置信分（考虑图连通性和历史信任）

        Args:
            payloads:         所有上传的 NodePayload 列表
            accepted_ids:     通过 Byzantine 检测的 Client ID 列表
            topology_scores:  各 Client 的拓扑置信分

        Returns:
            (global_embedding, aggregated_w, aggregated_b):
              global_embedding: [D]    全局语义中心
              aggregated_w:     [D, C] 全局分类器权重
              aggregated_b:     [C]    全局分类器偏置
        """
        print(f"\n  [Server] 执行拓扑感知 FedAvg | 参与 Client 数={len(accepted_ids)}")

        payload_map: Dict[int, NodePayload] = {
            p.client_id: p for p in payloads
        }

        # ── 计算加权系数 ──────────────────────────────────────────
        # weight_k = n_k × s_k
        weights: Dict[int, float] = {}
        total_weight: float = 0.0

        for client_id in accepted_ids:
            payload = payload_map[client_id]
            s_k = topology_scores[client_id]
            n_k = float(payload.num_samples)
            w_k = n_k * s_k
            weights[client_id] = w_k
            total_weight += w_k

        if total_weight < 1e-8:
            # 防御性处理：若所有权重为 0，退化为均匀权重
            total_weight = float(len(accepted_ids))
            weights = {cid: 1.0 for cid in accepted_ids}

        # ── 加权聚合全局嵌入中心 ──────────────────────────────────
        # global_embedding = Σ_k ( w_k / W_total ) × mean(H_k)
        global_embedding = torch.zeros(self.hidden_dim)     # [D]
        aggregated_w = torch.zeros(self.hidden_dim, self.num_classes)   # [D, C]
        aggregated_b = torch.zeros(self.num_classes)        # [C]

        for client_id in accepted_ids:
            payload = payload_map[client_id]
            normalized_weight: float = weights[client_id] / total_weight

            # 聚合嵌入中心
            # payload.embeddings: [N_k, D] → mean(dim=0): [D]
            client_center = payload.embeddings.mean(dim=0)  # [D]
            global_embedding += normalized_weight * client_center  # [D]

            # 聚合分类器权重
            # payload.classifier_w: [D, C]
            aggregated_w += normalized_weight * payload.classifier_w   # [D, C]

            # 聚合分类器偏置
            # payload.classifier_b: [C]
            aggregated_b += normalized_weight * payload.classifier_b   # [C]

            print(
                f"    [Server] Client {client_id} | 归一化权重={normalized_weight:.3f} | "
                f"拓扑分={topology_scores[client_id]:.3f}"
            )

        # 最终全局嵌入 L2 归一化
        global_embedding = F.normalize(global_embedding, p=2, dim=0)   # [D]

        return global_embedding, aggregated_w, aggregated_b

    # ─────────────────────────────────────────────────────────────
    # 核心方法 4: 主聚合入口（每轮调用一次）
    # ─────────────────────────────────────────────────────────────
    def aggregate(
        self,
        payloads: List[NodePayload],
    ) -> GlobalPayload:
        """
        【SERVER-SIDE 主计算】执行一轮完整的拓扑安全聚合，返回 GlobalPayload。

        聚合流程：
          1. Byzantine 检测（余弦相似度 + 域漂移过滤）
          2. 拓扑置信分计算（基于子图连通度）
          3. 拓扑感知 FedAvg 聚合（仅使用通过检测的 Client）
          4. 构建并返回 GlobalPayload

        Args:
            payloads: 本轮所有 Client 上传的 NodePayload 列表

        Returns:
            global_payload: 广播给所有 Client 的全局更新载荷
        """
        print(f"\n{'='*60}")
        print(f"[Server] ===== Round {self.round_id} 聚合开始 =====")
        print(f"[Server] 接收到 {len(payloads)} 个 Client 的 Payload")
        print(f"{'='*60}")

        # ── Phase 1: Byzantine 检测（SERVER-ONLY 安全计算）────────
        accepted_ids, anomaly_ids = self._detect_byzantine_nodes(payloads)

        if len(accepted_ids) == 0:
            raise RuntimeError(
                f"Round {self.round_id}: 所有 Client 均被标记为异常！"
                f"检查 byzantine_threshold 配置或 Client 数据质量。"
            )

        # ── Phase 2: 拓扑置信分计算（SERVER-ONLY）────────────────
        topology_scores = self._compute_topology_scores(payloads, accepted_ids)

        # ── Phase 3: 拓扑感知 FedAvg 聚合（SERVER-ONLY）─────────
        global_embedding, aggregated_w, aggregated_b = (
            self._topology_aware_fedavg(payloads, accepted_ids, topology_scores)
        )

        # ── Phase 4: 构建 GlobalPayload ───────────────────────────
        global_payload = GlobalPayload(
            round_id=self.round_id,
            global_embedding=global_embedding,      # [D]
            aggregated_w=aggregated_w,              # [D, C]
            aggregated_b=aggregated_b,              # [C]
            topology_scores=topology_scores,
            anomaly_flags=anomaly_ids,
            accepted_clients=accepted_ids,
        )

        # ── 更新 Server 状态 ──────────────────────────────────────
        self.last_global_w = aggregated_w.clone()   # [D, C]  保存历史权重
        self.round_id += 1                          # 轮次计数器递增

        print(f"\n[Server] Round {self.round_id - 1} 聚合完成")
        print(f"  ✅ 有效 Client: {accepted_ids}")
        print(f"  ❌ 异常 Client: {anomaly_ids}")
        print(f"  全局嵌入 L2 范数: {global_embedding.norm().item():.4f}")
        print(f"  全局分类器权重范数: {aggregated_w.norm().item():.4f}")

        return global_payload

    # ─────────────────────────────────────────────────────────────
    # 评估方法：计算全局模型分类准确率（使用聚合后的分类器）
    # ─────────────────────────────────────────────────────────────
    def evaluate_global_model(
        self,
        global_payload: GlobalPayload,
        test_embeddings: torch.Tensor,  # shape: [M, D]
        test_labels: torch.Tensor,      # shape: [M]
    ) -> Dict[str, float]:
        """
        【SERVER-SIDE 评估】用全局聚合分类器评估模型性能。

        Args:
            global_payload:   当前轮的全局聚合载荷
            test_embeddings:  测试集嵌入，shape = [M, D]
            test_labels:      测试集标签，shape = [M]

        Returns:
            metrics: 包含 accuracy、loss 的评估字典
        """
        # global_payload.aggregated_w shape: [D, C]
        # test_embeddings shape: [M, D]
        # [M, D] @ [D, C] → [M, C]
        logits = test_embeddings @ global_payload.aggregated_w  # [M, C]
        logits = logits + global_payload.aggregated_b.unsqueeze(0)  # [M, C]

        preds = logits.argmax(dim=-1)                   # [M]
        accuracy = (preds == test_labels).float().mean().item()

        loss = F.cross_entropy(logits, test_labels).item()

        metrics = {
            "round": global_payload.round_id,
            "accuracy": accuracy,
            "loss": loss,
            "accepted_clients": len(global_payload.accepted_clients),
            "anomaly_clients": len(global_payload.anomaly_flags),
        }
        return metrics

    # ─────────────────────────────────────────────────────────────
    # 查询方法：获取 Server 状态摘要
    # ─────────────────────────────────────────────────────────────
    def get_server_status(self) -> dict:
        """返回 Server 状态摘要（用于监控，不包含敏感聚合数据）。"""
        return {
            "round_id": self.round_id,
            "client_trust_scores": dict(self.client_trust_scores),
            "has_global_center": self.global_embedding_center is not None,
            "byzantine_threshold": self.byzantine_threshold,
            "domain_shift_max": self.domain_shift_max,
        }
