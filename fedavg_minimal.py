"""
fedavg_minimal.py — 极简手写 FedAvg 联邦学习（Java 后端视角全注释版）
═════════════════════════════════════════════════════════════════════
【学习目标】
不依赖 Flower、PySyft 等任何重量级第三方联邦框架，纯手写 PyTorch 实现 FedAvg。
从 Java 分布式/微服务视角理解联邦学习的核心生命周期：
  1. Server (服务端/注册中心) 广播全局模型权重快照 (load_state_dict)
  2. Client (分布式客户端节点) 接收快照，利用本地私有数据隔离训练 (无共享内存，数据不出本地)
  3. Server 接收各节点的参数字典 (Map<String, Tensor>)，执行按样本量加权平均 (FedAvg 核心算法)
  4. 循环迭代，直至收敛。

运行命令：
  /home/yix/torch_venv/bin/python fedavg_minimal.py
"""

from __future__ import annotations

import copy
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═════════════════════════════════════════════════════════════════
# 第一部分：定义模型实体 (Entity / Model)
# ═════════════════════════════════════════════════════════════════
class SimpleMLP(nn.Module):
    """
    多层感知机 (Multi-Layer Perceptron, 简称 MLP)。
    在 Java 视角中，它相当于一个包含了特征计算逻辑的 Domain 实体。
    
    网络结构设计（漏斗型降维）：
      输入特征 (20 维) ──► 隐藏层 (16 维) + ReLU 激活 ──► 输出分类得分 (2 维)
      
    为什么降维？
      - 20 -> 16: 信息压缩与特征抽象，去除冗余特征与噪声。
      - 16 -> 2:  任务收敛，因为当前是二分类任务 (例如: 0-正常, 1-异常)，输出必须是类别数 2。
    """
    def __init__(self, in_features: int = 20, hidden_dim: int = 16, num_classes: int = 2) -> None:
        super().__init__()
        # 第一层全连接层 (Fully Connected, fc1): W1 的形状是 [16, 20], b1 是 [16]
        # 相当于线性方程: y = x @ W1.T + b1
        self.fc1 = nn.Linear(in_features, hidden_dim)

        # 第二层输出层 (fc2): W2 的形状是 [2, 16], b2 是 [2]
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播链路 (Forward Pass)。
        注意：深度学习中，层与层之间的连接必须保持“数学可导性”，不能随意插入不可导的 Java 式业务代码，
        否则反向传播 (Autograd) 无法通过链式法则计算梯度。

        Args:
            x: 输入特征批次，Tensor Shape: [Batch_Size, 20]

        Returns:
            logits: 未经 Softmax 归一化的类别得分，Tensor Shape: [Batch_Size, 2]
        """
        # 1. 经过第一层矩阵乘法: [B, 20] @ [20, 16] -> [B, 16]
        out = self.fc1(x)                               # Shape: [B, 16]

        # 2. 激活函数 ReLU (非线性过滤，类似 if (x < 0) x = 0; 但处处平滑且梯度容易计算)
        out = F.relu(out)                               # Shape: [B, 16]

        # 3. 经过第二层分类头映射: [B, 16] @ [16, 2] -> [B, 2]
        logits = self.fc2(out)                          # Shape: [B, 2]
        
        return logits                                   # Shape: [B, 2]


# ═════════════════════════════════════════════════════════════════
# 第二部分：客户端 (Client)
# ═════════════════════════════════════════════════════════════════
class Client:
    """
    代表一个独立的边缘节点或机构（例如：一家医院、一个部门）。
    
    【核心设计原则】：
    1. 拥有完全独立的本地私有数据库 (data_X, data_y)，绝不暴露给外界，外界也绝不能调用它。
    2. 拥有独立的本地模型 (local_model)，内存与 Server 必须彻底解耦。
    """
    def __init__(self, client_id: int, num_samples: int) -> None:
        self.client_id = client_id

        # ─────────────────────────────────────────────────────────
        # 1. 本地私有数据（模拟数据库中的持久化数据）
        # ─────────────────────────────────────────────────────────
        # data_X 模拟特征矩阵: [num_samples, 20]
        # 相当于 Java 中查出的 List<UserFeatureDTO>，转成了浮点数张量
        self.data_X: torch.Tensor = torch.randn(num_samples, 20)           # Shape: [N, 20]

        # data_y 模拟标签向量: [num_samples] 取值为 0 或 1
        # 相当于真实的业务标注 (如是否患病、是否为欺诈交易)
        self.data_y: torch.Tensor = torch.randint(0, 2, (num_samples,))    # Shape: [N]

        # ─────────────────────────────────────────────────────────
        # 2. 本地模型副本
        # ─────────────────────────────────────────────────────────
        self.local_model: SimpleMLP = SimpleMLP()

    def receive_global_model(self, global_model: nn.Module) -> None:
        """
        【联邦流程第 ① 步：接收全局模型快照】
        
        相当于 Server 通过 RPC 广播配置，Client 收到后覆盖本地配置。
        
        关键注意点：
        必须使用 copy.deepcopy()！
        如果直接写 self.local_model.load_state_dict(global_model.state_dict())，
        PyTorch 的底层张量可能仍指向同一个底层存储指针，会导致 Client 训练时直接原地修改 Server 的内存，
        破坏分布式物理隔离的假定。
        """
        # state_dict() 本质上就是一个 Map<String, Tensor>
        # key 是层名称 (如 'fc1.weight', 'fc1.bias')，value 是具体的权重张量
        server_state: Dict[str, torch.Tensor] = copy.deepcopy(global_model.state_dict())
        self.local_model.load_state_dict(server_state)

    def local_train(self, num_epochs: int = 3, lr: float = 0.01) -> float:
        """
        【联邦流程第 ② 步：本地数据训练】
        
        在完全隔离的环境下，用私有数据更新 local_model 的参数。
        数据 (data_X, data_y) 始终停留在本地内存中，绝不跨网络传输。
        
        Args:
            num_epochs: 本地迭代轮数 (联邦学习通常为 1~5 轮，避免本地跑得过远导致漂移)
            lr: 学习率 (步长)
            
        Returns:
            最终损失值 (仅用于本地打日志打印)
        """
        # 声明针对本节点 local_model 的优化器 (SGD 随机梯度下降)
        # 严禁使用 server.global_model.parameters()，因为优化器更新的是它绑定的模型参数
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr)

        final_loss = 0.0
        for epoch in range(num_epochs):
            optimizer.zero_grad()                         # 清除上一步残留的梯度 (每次反向传播前必须清零)
            
            # 前向计算: [N, 20] -> [N, 2]
            logits = self.local_model(self.data_X)        # Shape: [N, 2]
            
            # 计算交叉熵损失 (衡量模型预测与真实标签之间的差距)
            loss = F.cross_entropy(logits, self.data_y)   # Scalar 标量
            
            # 反向传播: 计算 loss 对每个可学习参数的偏导数 (梯度)
            loss.backward()
            
            # 权重更新: W = W - lr * grad
            optimizer.step()

            final_loss = loss.item()

        return final_loss

    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """
        【联邦流程第 ③ 步：参数序列化上报】
        
        生成准备发送给 Server 的 Payload (DTO)。
        
        安全边界说明：
        返回的字典里只有权重张量 (W, b)，绝对不包含：
        - 原始特征 data_X
        - 原始标签 data_y
        - 过程梯度与中间激活值
        """
        return copy.deepcopy(self.local_model.state_dict())

    @property
    def num_samples(self) -> int:
        """返回本地样本数量 N，用于后续 FedAvg 计算加权系数。"""
        return len(self.data_X)


# ═════════════════════════════════════════════════════════════════
# 第三部分：中心协调服务器 (Server)
# ═════════════════════════════════════════════════════════════════
class Server:
    """
    负责聚合全局模型的协调者。
    
    【核心职责】：
    1. 维护唯一的全局基线模型 (global_model)。
    2. 接收各 Client 上报的模型权重列表。
    3. 执行加权平均算法 (FedAvg)。
    4. 评估全局模型的性能。
    """
    def __init__(self) -> None:
        self.global_model: SimpleMLP = SimpleMLP()

    def aggregate(
        self,
        client_params_list: List[Dict[str, torch.Tensor]],
        client_weights: List[int],
    ) -> None:
        """
        【FedAvg 核心算法实现】
        
        数学公式：
            W_global = Σ_k ( n_k / Σ n_i ) * W_k
            
        解释：
            - n_k 是第 k 个客户端的样本量。
            - 数据量越大的客户端，在加权平均中拥有的发言权 (权重) 越大。
            
        实现步骤：
            1. 拿到各客户端样本总和。
            2. 取一个参数字典作为模版，把所有张量初始化为 0。
            3. 遍历每个客户端的每个参数键值对，按比例累加。
            4. 将加权平均后的结果赋值回 global_model。
        """
        total_samples: int = sum(client_weights)          # 所有客户端的样本总量 Σ n_i

        # 1. 复制一份参数字典结构作为累加容器
        global_params: Dict[str, torch.Tensor] = copy.deepcopy(client_params_list[0])

        # 2. 将容器内所有张量全部清零，准备累加
        for key in global_params:
            global_params[key] = torch.zeros_like(global_params[key])

        # 3. 双层循环执行加权累加 (类似 Java 中对 Map 数据的归并操作)
        for client_params, n_k in zip(client_params_list, client_weights):
            weight_ratio = float(n_k) / float(total_samples)   # 归一化权重系数: n_k / Σ n_i

            for key in global_params:
                # 矩阵运算：累加 weight_ratio * client_W
                global_params[key] += weight_ratio * client_params[key]

        # 4. 把计算好的加权平均参数覆盖到全局模型中 (完成聚合)
        self.global_model.load_state_dict(global_params)

    def evaluate(self, num_test_samples: int = 100) -> float:
        """
        在全局测试集上验证当前聚合后模型的准确率。
        """
        test_X = torch.randn(num_test_samples, 20)          # Shape: [100, 20]
        test_y = torch.randint(0, 2, (num_test_samples,))   # Shape: [100]

        # 进入评估模式 (禁用 Dropout, BatchNorm 等行为)
        self.global_model.eval()

        with torch.no_grad():                               # 禁用自动求导，节省显存并加快推理
            logits = self.global_model(test_X)              # Shape: [100, 2]
            preds = logits.argmax(dim=-1)                   # 取最大概率索引作为预测类别，Shape: [100]
            accuracy = (preds == test_y).float().mean().item()

        return accuracy


# ═════════════════════════════════════════════════════════════════
# 第四部分：调度控制器 / 端到端主函数 (Main Loop)
# ═════════════════════════════════════════════════════════════════
def run_fedavg(
    num_clients: int = 3,
    num_rounds: int = 5,
    local_epochs: int = 3,
) -> None:
    """
    驱动整个联邦学习训练周期的调度入口。
    相当于微服务环境中的 Orchestrator / Workflow 引擎。
    """
    print("=" * 60)
    print("🚀 手写极简 FedAvg 联邦学习生命周期演示")
    print("=" * 60)

    # 1. 实例化 Server 与 Clients
    server = Server()
    
    # 模拟 3 个客户端，各自拥有不同规模的数据集 (20, 40, 60 个样本)
    clients = [
        Client(client_id=i, num_samples=(i + 1) * 20)
        for i in range(num_clients)
    ]

    print(f"参与客户端数: {num_clients}")
    print(f"客户端样本分布: {[c.num_samples for c in clients]} (数据倾斜/异构模拟)")
    print()

    # 2. 联邦训练多轮迭代循环
    for round_idx in range(num_rounds):
        print(f"┌─── Round {round_idx + 1}/{num_rounds} ───────────────────────────────────────")

        # ──【阶段 1：下发】Server 广播当前全局模型给每个 Client ──
        for client in clients:
            client.receive_global_model(server.global_model)

        # ──【阶段 2：训练】每个 Client 使用本地数据独立微调 ────
        for client in clients:
            loss = client.local_train(num_epochs=local_epochs)
            print(f"│  [Client {client.client_id}] 本地训练完成 | 样本量={client.num_samples} | 最终 Loss={loss:.4f}")

        # ──【阶段 3：上报】收集每个 Client 的脱敏权重字典 ──────
        all_params = [client.get_model_params() for client in clients]
        all_sample_counts = [client.num_samples for client in clients]

        # ──【阶段 4：聚合】Server 执行 FedAvg 加权平均 ──────────
        server.aggregate(all_params, all_sample_counts)

        # ──【阶段 5：验证】评估更新后的全局模型 ────────────────
        acc = server.evaluate()
        print(f"│  ► [Server] 聚合完成 | 全局模型准确率={acc * 100:.2f}%")
        print("└──────────────────────────────────────────────────────────\n")

    print("=" * 60)
    print("🏁 联邦训练圆满完成！")
    print("=" * 60)
    print("【总结复习 - 记住这四个关键映射】：")
    print(" 1. 数据隔离: data_X 存在 Client 内部，永远不进入传输网络。")
    print(" 2. 消息载荷: 跨网络只传 state_dict (Map<String, Tensor>)。")
    print(" 3. 聚合本质: 按样本数比例做字典累加平均。")
    print(" 4. 框架可替代性: 不管以后换成 GNN、Transformer 还是 ResNet，这套外层循环 100% 通用。")
    print("=" * 60)


if __name__ == "__main__":
    run_fedavg(num_clients=3, num_rounds=5, local_epochs=3)
