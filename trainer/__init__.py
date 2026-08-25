from .ppo_trainer import PPOTrainer
from .q_stra_trainer import StrategyQTrainer

# Backward-compatible alias for the original ToSCA entrypoint.
SFTTrainer = StrategyQTrainer
