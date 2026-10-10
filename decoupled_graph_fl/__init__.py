"""
__init__.py
═══════════════════════════════════════════════════════════════════
decoupled_graph_fl 包初始化文件
对外暴露核心组件，简化导入路径。
"""

from .frozen_encoder import FrozenEncoder, VocabPartitionConfig
from .payloads import GlobalPayload, NodePayload
from .federated_graph_node import FederatedGraphNode
from .topology_coordinator import TopologyCoordinator

__all__ = [
    "FrozenEncoder",
    "VocabPartitionConfig",
    "NodePayload",
    "GlobalPayload",
    "FederatedGraphNode",
    "TopologyCoordinator",
]

__version__ = "0.1.0"
__description__ = "Decoupled Graph Federated Learning Topology Security Framework"
