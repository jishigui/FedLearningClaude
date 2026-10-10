"""
fedavg_minimal.py — 最小可运行的 FedAvg 联邦学习
══════════════════════════════════════════════════
【学习目标】在不使用任何联邦学习框架的前提下，
用最少的代码搞清楚 FedAvg 的核心三步：

  1. Server 把全局模型发给每个 Client
  2. 每个 Client 用自己的本地数据训练几步
  3. Server 把所有 Client 的模型参数加权平均，更新全局模型

就这三步，循环 N 轮，就是 FedAvg。

运行方法：
  /home/yix/torch_venv/bin/python fedavg_minimal.py
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════
# 第一部分：定义一个最简单的神经网络
# （联邦学习跟网络结构无关，这里用最简单的 MLP）
# ════════════════════════════════════════════════
class SimpleMLP(nn.Module):
    """两层 MLP，输入 20 维特征，输出 2 类。"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 16)   # [B, 20] → [B, 16]
        self.fc2 = nn.Linear(16,  2)   # [B, 16] → [B,  2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))         # [B, 16]
        return self.fc2(x)              # [B,  2]


# ════════════════════════════════════════════════
# 第二部分：Client（客户端）
# 每个 Client 代表一个数据拥有方（例如一家医院）
# ════════════════════════════════════════════════
class Client:
    def __init__(self, client_id: int, num_samples: int):
        self.client_id = client_id

        # 每个 Client 有自己的私有数据（绝不上传！）
        # 这里用随机数模拟，真实场景是各自的本地数据集
        self.data_X = torch.randn(num_samples, 20)          # [N, 20] 特征
        self.data_y = torch.randint(0, 2, (num_samples,))   # [N]     标签

        # Client 本地持有一个模型副本（从 Server 下发）
        self.local_model = SimpleMLP()

    def receive_global_model(self, global_model: nn.Module):
        """
        【联邦学习第一步】接收 Server 发来的全局模型，
        用它覆盖本地模型（深拷贝，互不影响）。
        """
        self.local_model.load_state_dict(
            copy.deepcopy(global_model.state_dict())
        )

    def local_train(self, num_epochs: int = 3, lr: float = 0.01):
        """
        【联邦学习第二步】用本地私有数据训练几步。
        注意：训练只发生在本地，数据不出本地！
        """
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            logits = self.local_model(self.data_X)           # [N, 2]
            loss = F.cross_entropy(logits, self.data_y)      # scalar
            loss.backward()
            optimizer.step()

        # 返回最终 loss 用于打印（不是必须的）
        return loss.item()

    def get_model_params(self) -> dict:
        """
        上传给 Server 的东西：只有模型参数（权重），
        原始数据 self.data_X / self.data_y 永远留在本地。
        """
        return copy.deepcopy(self.local_model.state_dict())

    @property
    def num_samples(self) -> int:
        return len(self.data_X)


# ════════════════════════════════════════════════
# 第三部分：Server（服务器）
# Server 不持有任何客户端数据，只做模型聚合
# ════════════════════════════════════════════════
class Server:
    def __init__(self):
        # 全局模型：Server 维护的唯一模型
        self.global_model = SimpleMLP()

    def aggregate(self, client_params_list: list, client_weights: list):
        """
        【联邦学习第三步】FedAvg 加权平均。

        公式：w_global = Σ_k (n_k / Σn_k) × w_k

        client_params_list: 每个 Client 上传的模型参数字典
        client_weights:     每个 Client 的样本数（用于加权）
        """
        total_samples = sum(client_weights)         # Σ n_k

        # 初始化：把全局模型参数全部清零，准备累加
        global_params = copy.deepcopy(client_params_list[0])
        for key in global_params:
            global_params[key] = torch.zeros_like(global_params[key])

        # 加权求和：Σ (n_k / Σn_k) × w_k
        for params, n_k in zip(client_params_list, client_weights):
            weight = n_k / total_samples            # 归一化权重
            for key in global_params:
                # 每层参数加权累加
                global_params[key] += weight * params[key]

        # 用聚合后的参数更新全局模型
        self.global_model.load_state_dict(global_params)

    def evaluate(self) -> float:
        """用随机测试集评估全局模型准确率（模拟）。"""
        test_X = torch.randn(100, 20)
        test_y = torch.randint(0, 2, (100,))
        with torch.no_grad():
            logits = self.global_model(test_X)      # [100, 2]
            preds  = logits.argmax(dim=-1)          # [100]
            acc = (preds == test_y).float().mean().item()
        return acc


# ════════════════════════════════════════════════
# 第四部分：联邦训练主循环
# 把上面三个部分串起来，这就是完整的 FedAvg！
# ════════════════════════════════════════════════
def run_fedavg(num_clients=3, num_rounds=5, local_epochs=3):
    print("=" * 50)
    print("  最小 FedAvg 联邦学习演示")
    print("=" * 50)

    # 1. 创建 Server 和 Clients
    server  = Server()
    clients = [
        Client(client_id=i, num_samples=(i + 1) * 20)
        for i in range(num_clients)
    ]
    # 三个 Client 的样本数：20, 40, 60（不同 Client 数据量不同）

    print(f"客户端数量: {num_clients}")
    print(f"各客户端样本数: {[c.num_samples for c in clients]}")
    print()

    # 2. 主循环：每轮重复 3 步
    for round_idx in range(num_rounds):
        print(f"── Round {round_idx + 1}/{num_rounds} ──────────────")

        # 【第一步】Server → Client：下发全局模型
        for client in clients:
            client.receive_global_model(server.global_model)

        # 【第二步】每个 Client 本地训练
        for client in clients:
            loss = client.local_train(num_epochs=local_epochs)
            print(f"  Client {client.client_id}: loss={loss:.4f}")

        # 【第三步】Client → Server：上传模型参数，Server 聚合
        all_params  = [c.get_model_params() for c in clients]
        all_weights = [c.num_samples        for c in clients]
        server.aggregate(all_params, all_weights)

        # 评估全局模型
        acc = server.evaluate()
        print(f"  → 全局模型准确率: {acc:.3f}")
        print()

    print("训练完成！")
    print()
    print("━" * 50)
    print("核心理解总结：")
    print("  Client 有：私有数据 + 本地模型副本")
    print("  Server 有：全局模型（无任何客户端数据）")
    print("  每轮传输：全局模型权重（下发）+ 本地模型权重（上传）")
    print("  聚合方式：按样本数加权平均（FedAvg）")
    print("━" * 50)


if __name__ == "__main__":
    run_fedavg(num_clients=3, num_rounds=5, local_epochs=3)

