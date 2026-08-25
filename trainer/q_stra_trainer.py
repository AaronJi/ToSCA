from abc import ABC
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler
from tqdm import tqdm


class StrategyQTrainer(ABC):
    """High-level ToSCA trainer.

    The ToSCA paper defines Q_H(s, a) as the averaged language-model score of
    the textual strategy action appended to the high-level prompt. This trainer
    optimizes that value with the DQN target:

        r_sat + gamma * max_a' Q_target(s', a')
    """

    def __init__(
        self,
        model,
        target_model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        max_norm: float = 1,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        action_tokens: Sequence[str] = None,
        target_update_steps: int = 10,
        gamma: float = 0.85,
        q_value_mode: str = "logit",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.model = model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.target_update_steps = target_update_steps
        self.gamma = gamma
        self.q_value_mode = q_value_mode

        if not action_tokens:
            raise ValueError("StrategyQTrainer requires non-empty action_tokens.")
        self.action_tokens = tuple(action_tokens)
        self.action_token_ids = [
            self.tokenizer(token, add_special_tokens=False)["input_ids"] for token in self.action_tokens
        ]
        if any(len(token_ids) == 0 for token_ids in self.action_token_ids):
            raise ValueError(f"Could not tokenize one or more strategy action tokens: {self.action_tokens}")

        self._wandb = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)

    def fit(self, args):
        if args.eval_steps == -1:
            args.eval_steps = self.train_dataloader.__len__()
        if args.save_steps == -1:
            args.save_steps = float("inf")

        self._sync_target_model()
        global_step = 1
        epoch_bar = tqdm(range(self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())

        for epoch in range(self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(epoch)

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc="Train step of epoch %d" % epoch,
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            self.target_model.eval()
            loss_mean = 0.0

            for batch in self.train_dataloader:
                logs_dict = self.training_step(batch)

                if self.target_update_steps > 0 and global_step % self.target_update_steps == 0:
                    self._sync_target_model()

                loss_mean = loss_mean * 0.95 + 0.05 * logs_dict["q_loss"]
                logs_dict["loss_mean"] = loss_mean
                self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict)

                step_bar.update()
                global_step += 1

            epoch_bar.update()

    def training_step(self, batch) -> Dict[str, float]:
        (
            current_input_ids,
            current_attention_masks,
            next_input_ids,
            next_attention_masks,
            strategy_indices,
            rewards,
            dones,
        ) = batch

        device = torch.cuda.current_device()
        current_input_ids = current_input_ids.to(device)
        current_attention_masks = current_attention_masks.to(device)
        next_input_ids = next_input_ids.to(device)
        next_attention_masks = next_attention_masks.to(device)
        strategy_indices = strategy_indices.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)

        q_all = self._action_values(self.model, current_input_ids, current_attention_masks)
        q_current = q_all.gather(dim=1, index=strategy_indices.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next_all = self._action_values(self.target_model, next_input_ids, next_attention_masks)
            max_q_next = q_next_all.max(dim=1).values
            if getattr(self.args, "learn_from_org", False):
                rewards = torch.ones_like(rewards)
            q_targets = rewards + (1.0 - dones) * self.gamma * max_q_next

        q_loss = F.mse_loss(q_current, q_targets)
        self.strategy.backward(q_loss, self.model, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

        pred_indices = q_all.detach().argmax(dim=1)
        accuracy = (pred_indices == strategy_indices).float().mean()
        return {
            "q_loss": q_loss.item(),
            "q": q_current.detach().mean().item(),
            "q_target": q_targets.detach().mean().item(),
            "reward": rewards.detach().mean().item(),
            "strategy_acc": accuracy.item(),
        }

    def _action_values(self, actor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if all(len(token_ids) == 1 for token_ids in self.action_token_ids):
            return self._single_token_action_values(actor, input_ids, attention_mask)
        return self._multi_token_action_values(actor, input_ids, attention_mask)

    def _single_token_action_values(
        self,
        actor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._forward_logits(actor, input_ids, attention_mask)
        if self.q_value_mode == "logprob":
            logits = F.log_softmax(logits, dim=-1)
        elif self.q_value_mode == "prob":
            logits = F.softmax(logits, dim=-1)

        last_valid_index = attention_mask.long().sum(dim=1) - 1
        batch_indices = torch.arange(input_ids.shape[0], device=input_ids.device)
        last_logits = logits[batch_indices, last_valid_index, :]
        action_ids = torch.tensor(
            [token_ids[0] for token_ids in self.action_token_ids],
            dtype=torch.long,
            device=input_ids.device,
        )
        return last_logits.index_select(dim=1, index=action_ids)

    def _multi_token_action_values(
        self,
        actor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        values = []
        for token_ids in self.action_token_ids:
            values.append(self._multi_token_value(actor, input_ids, attention_mask, token_ids))
        return torch.stack(values, dim=1)

    def _multi_token_value(
        self,
        actor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_token_ids: Sequence[int],
    ) -> torch.Tensor:
        device = input_ids.device
        action_tensor = torch.tensor(action_token_ids, dtype=input_ids.dtype, device=device)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0

        sequences = []
        masks = []
        prompt_lengths = []
        for row in range(input_ids.size(0)):
            prompt_len = int(attention_mask[row].long().sum().item())
            prompt_lengths.append(prompt_len)
            prompt_tokens = input_ids[row, :prompt_len]
            sequence = torch.cat((prompt_tokens, action_tensor))
            sequences.append(sequence)
            masks.append(torch.ones_like(sequence))

        max_len = max(sequence.size(0) for sequence in sequences)
        padded_sequences = []
        padded_masks = []
        for sequence, mask in zip(sequences, masks):
            pad_len = max_len - sequence.size(0)
            padded_sequences.append(F.pad(sequence, (0, pad_len), value=pad_token_id))
            padded_masks.append(F.pad(mask, (0, pad_len), value=0))

        batch_ids = torch.stack(padded_sequences, dim=0)
        batch_masks = torch.stack(padded_masks, dim=0)
        logits = self._forward_logits(actor, batch_ids, batch_masks)
        if self.q_value_mode == "logprob":
            logits = F.log_softmax(logits, dim=-1)
        elif self.q_value_mode == "prob":
            logits = F.softmax(logits, dim=-1)

        token_values = []
        for offset, token_id in enumerate(action_token_ids):
            positions = torch.tensor(
                [prompt_len - 1 + offset for prompt_len in prompt_lengths],
                dtype=torch.long,
                device=device,
            )
            rows = torch.arange(input_ids.size(0), device=device)
            token_values.append(logits[rows, positions, int(token_id)])
        return torch.stack(token_values, dim=1).mean(dim=1)

    def _forward_logits(self, actor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = actor(input_ids, attention_mask=attention_mask, return_output=True)
        if isinstance(output, tuple):
            output = output[-1]
        return output.logits if hasattr(output, "logits") else output["logits"]

    def _sync_target_model(self) -> None:
        source = self._unwrap_actor_model(self.model)
        target = self._unwrap_actor_model(self.target_model)
        if getattr(self.strategy, "stage", 0) != 3:
            target.load_state_dict(source.state_dict(), strict=False)
            return

        import deepspeed

        from openrlhf.utils.deepspeed_utils import _z3_params_to_fetch

        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                params_to_fetch = _z3_params_to_fetch([source_param, target_param])
                with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                    target_param.data.copy_(source_param.data)

            for source_buffer, target_buffer in zip(source.buffers(), target.buffers()):
                target_buffer.data.copy_(source_buffer.data)

    @staticmethod
    def _unwrap_actor_model(actor):
        model = actor.model if hasattr(actor, "model") else actor
        return model.module if hasattr(model, "module") else model

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict=None):
        logs_dict = logs_dict or {}
        if global_step % args.logging_steps == 0:
            logs_dict = self.strategy.all_reduce(logs_dict)
            step_bar.set_postfix(logs_dict)

            if (
                self._wandb is not None
                and self.strategy.is_rank_0()
                and global_step % self.strategy.accumulated_gradient == 0
            ):
                logs = {"train/%s" % k: v for k, v in {**logs_dict, "global_step": global_step}.items()}
                self._wandb.log(logs)

        if self.eval_dataloader is not None and global_step % args.eval_steps == 0:
            eval_logs = self.evaluate(self.eval_dataloader, global_step)
            if self._wandb is not None and self.strategy.is_rank_0():
                self._wandb.log({f"eval/{k}": v for k, v in eval_logs.items()})

        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem)

    def evaluate(self, eval_dataloader, steps=0) -> Dict[str, float]:
        self.model.eval()
        totals = {"eval_q_loss": 0.0, "eval_strategy_acc": 0.0}
        count = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                (
                    current_input_ids,
                    current_attention_masks,
                    next_input_ids,
                    next_attention_masks,
                    strategy_indices,
                    rewards,
                    dones,
                ) = batch
                device = torch.cuda.current_device()
                current_input_ids = current_input_ids.to(device)
                current_attention_masks = current_attention_masks.to(device)
                next_input_ids = next_input_ids.to(device)
                next_attention_masks = next_attention_masks.to(device)
                strategy_indices = strategy_indices.to(device)
                rewards = rewards.to(device)
                dones = dones.to(device)

                q_all = self._action_values(self.model, current_input_ids, current_attention_masks)
                q_current = q_all.gather(dim=1, index=strategy_indices.unsqueeze(1)).squeeze(1)
                q_next_all = self._action_values(self.target_model, next_input_ids, next_attention_masks)
                q_targets = rewards + (1.0 - dones) * self.gamma * q_next_all.max(dim=1).values
                totals["eval_q_loss"] += F.mse_loss(q_current, q_targets).item()
                totals["eval_strategy_acc"] += (q_all.argmax(dim=1) == strategy_indices).float().mean().item()
                count += 1

        self.model.train()
        if count == 0:
            return {"global_step": steps, **totals}
        return {"global_step": steps, **{key: value / count for key, value in totals.items()}}
