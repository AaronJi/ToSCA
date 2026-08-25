from typing import Any, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from openrlhf.utils.tosca import (
    build_high_level_prompt,
    build_low_level_prompt,
    resolve_action_tokens,
    resolve_strategies,
)

from .utils import zero_pad_sequences


def _first_existing(sample: Mapping[str, Any], keys: Sequence[Optional[str]]) -> Optional[Any]:
    for key in keys:
        if key and key in sample and sample[key] is not None:
            return sample[key]
    return None


def _has_prefixed_state(sample: Mapping[str, Any], prefix: str) -> bool:
    return any(
        f"{prefix}{key}" in sample
        for key in (
            "emotion",
            "emo",
            "description",
            "desc",
            "history",
            "hist",
            "query",
            "current_query",
        )
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "done", "terminal"}
    return False


class StrategyQDataset(Dataset):
    """Dataset for ToSCA high-level DQN training.

    Each item represents (s_t, a_t, r_sat_t, s_{t+1}, done). The current and
    next states can be provided either as ready-made prompts or as structured
    dialogue fields used by the ToSCA paper templates.
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        max_length: int,
        strategy,
        input_template: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.max_length = max_length
        self.input_template = input_template
        self.strategy_names = resolve_strategies(strategy.args)
        self.action_tokens = resolve_action_tokens(strategy.args, len(self.strategy_names))

        self.current_prompts = []
        self.next_prompts = []
        self.strategy_indices = []
        self.rewards = []
        self.dones = []

        for sample in tqdm(dataset, desc="Preprocessing ToSCA high-level data", disable=not strategy.is_rank_0()):
            current_prompt = self._build_current_prompt(sample)
            next_prompt, done = self._build_next_prompt(sample, current_prompt)
            strategy_index = self._strategy_to_index(sample)
            reward = self._reward(sample)
            done_key = getattr(strategy.args, "done_key", "done")
            done_value = _first_existing(sample, (done_key, "done", "terminal", "is_terminal"))
            if done_value is not None:
                done = _as_bool(done_value)

            if not current_prompt:
                continue
            self.current_prompts.append(current_prompt)
            self.next_prompts.append(next_prompt or current_prompt)
            self.strategy_indices.append(strategy_index)
            self.rewards.append(float(reward))
            self.dones.append(bool(done))

    def _build_current_prompt(self, sample: Mapping[str, Any]) -> str:
        args = self.strategy.args
        prompt = _first_existing(
            sample,
            (
                getattr(args, "current_input_key", None),
                "current_input",
                "current_prompt",
                "prompt",
                getattr(args, "input_key", "input"),
            ),
        )
        if prompt is None:
            return build_high_level_prompt(sample, self.strategy_names, self.action_tokens)
        prompt = str(prompt)
        if self.input_template:
            prompt = self.input_template.format(prompt)
        return prompt

    def _build_next_prompt(self, sample: Mapping[str, Any], current_prompt: str) -> tuple[str, bool]:
        args = self.strategy.args
        prompt = _first_existing(
            sample,
            (
                getattr(args, "next_input_key", None),
                "next_input",
                "next_prompt",
                "next_state",
            ),
        )
        if prompt is not None:
            return str(prompt), False
        prefix = getattr(args, "next_state_prefix", "next_")
        if _has_prefixed_state(sample, prefix):
            return build_high_level_prompt(sample, self.strategy_names, self.action_tokens, prefix=prefix), False
        return current_prompt, True

    def _strategy_to_index(self, sample: Mapping[str, Any]) -> int:
        args = self.strategy.args
        key = getattr(args, "strategy_key", "strategy")
        value = _first_existing(sample, (key, "strategy", "stra", "action", "label"))
        if value is None:
            raise ValueError("A ToSCA high-level sample is missing a strategy/action field.")

        if isinstance(value, torch.Tensor):
            value = value.item()

        if isinstance(value, int):
            if 0 <= value < len(self.strategy_names):
                return value
            if 1 <= value <= len(self.strategy_names):
                return value - 1

        value_text = str(value).strip()
        numeric = value_text.strip("()# ")
        if numeric.isdigit():
            numeric_value = int(numeric)
            if 1 <= numeric_value <= len(self.strategy_names):
                return numeric_value - 1
            if 0 <= numeric_value < len(self.strategy_names):
                return numeric_value

        lowered = value_text.lower()
        for index, token in enumerate(self.action_tokens):
            if lowered == token.lower():
                return index
        for index, name in enumerate(self.strategy_names):
            if lowered == name.lower():
                return index

        raise ValueError(f"Unknown strategy value: {value_text}")

    def _reward(self, sample: Mapping[str, Any]) -> float:
        args = self.strategy.args
        key = getattr(args, "reward_key", "reward")
        value = _first_existing(sample, (key, "reward", "rsat", "satisfaction", "score"))
        if value is None:
            return 1.0 if getattr(args, "learn_from_org", False) else 0.0
        if isinstance(value, torch.Tensor):
            value = value.item()
        return float(value)

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        return tokenized["input_ids"][0], tokenized["attention_mask"][0]

    def __len__(self) -> int:
        return len(self.current_prompts)

    def __getitem__(self, idx):
        current_ids, current_mask = self._tokenize(self.current_prompts[idx])
        next_ids, next_mask = self._tokenize(self.next_prompts[idx])
        return (
            current_ids,
            current_mask,
            next_ids,
            next_mask,
            torch.tensor(self.strategy_indices[idx], dtype=torch.long),
            torch.tensor(self.rewards[idx], dtype=torch.float),
            torch.tensor(self.dones[idx], dtype=torch.float),
        )

    def collate_fn(self, item_list):
        current_ids, current_masks, next_ids, next_masks, strategy_indices, rewards, dones = zip(*item_list)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
        return (
            zero_pad_sequences(current_ids, "right", pad_token_id),
            zero_pad_sequences(current_masks, "right", 0),
            zero_pad_sequences(next_ids, "right", pad_token_id),
            zero_pad_sequences(next_masks, "right", 0),
            torch.stack(strategy_indices),
            torch.stack(rewards),
            torch.stack(dones),
        )


class ToSCAPromptDataset(Dataset):
    """Prompt dataset that renders the low-level ToSCA response prompt."""

    def __init__(self, dataset, tokenizer, strategy) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.strategy_names = resolve_strategies(strategy.args)
        self.n_samples_per_prompt = getattr(strategy.args, "n_samples_per_prompt", 1)
        self.prompts = []

        for sample in tqdm(dataset, desc="Preprocessing ToSCA low-level prompts", disable=not strategy.is_rank_0()):
            strategy_name = self._strategy_name(sample)
            self.prompts.append(build_low_level_prompt(sample, strategy_name))

    def _strategy_name(self, sample: Mapping[str, Any]) -> str:
        key = getattr(self.strategy.args, "strategy_key", "strategy")
        value = _first_existing(sample, (key, "strategy", "stra", "action", "label"))
        if value is None:
            value = _first_existing(sample, ("selected_strategy", "predicted_strategy"))
        if value is None:
            raise ValueError("A ToSCA low-level prompt sample is missing a strategy field.")
        if isinstance(value, torch.Tensor):
            value = value.item()
        if isinstance(value, int):
            if 0 <= value < len(self.strategy_names):
                return self.strategy_names[value]
            if 1 <= value <= len(self.strategy_names):
                return self.strategy_names[value - 1]
        text = str(value).strip()
        numeric = text.strip("()# ")
        if numeric.isdigit():
            idx = int(numeric)
            if 1 <= idx <= len(self.strategy_names):
                return self.strategy_names[idx - 1]
            if 0 <= idx < len(self.strategy_names):
                return self.strategy_names[idx]
        return text

    def __len__(self) -> int:
        return len(self.prompts) * self.n_samples_per_prompt

    def __getitem__(self, idx):
        return self.prompts[idx // self.n_samples_per_prompt]
