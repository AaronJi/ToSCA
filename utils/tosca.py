from collections.abc import Mapping, Sequence
from typing import Any, Optional


DAILYDIALOG_STRATEGIES = (
    "Inform",
    "Question",
    "Directives",
    "Commissive",
)

ESCONV_STRATEGIES = (
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of Feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
)


def split_csv(value: Optional[str]) -> Optional[tuple[str, ...]]:
    if value is None:
        return None
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or None


def default_strategies(is_dailydialogue: bool = False) -> tuple[str, ...]:
    return DAILYDIALOG_STRATEGIES if is_dailydialogue else ESCONV_STRATEGIES


def default_action_tokens(num_strategies: int) -> tuple[str, ...]:
    return tuple(f"({idx})" for idx in range(1, num_strategies + 1))


def resolve_strategies(args) -> tuple[str, ...]:
    return split_csv(getattr(args, "strategies", None)) or default_strategies(
        getattr(args, "is_dailydialogue", False)
    )


def resolve_action_tokens(args, num_strategies: int) -> tuple[str, ...]:
    action_tokens = split_csv(getattr(args, "strategy_tokens", None)) or default_action_tokens(num_strategies)
    if len(action_tokens) != num_strategies:
        raise ValueError(
            f"Expected {num_strategies} strategy action tokens, got {len(action_tokens)}."
        )
    return action_tokens


def _value_from_aliases(
    sample: Mapping[str, Any],
    aliases: Sequence[str],
    *,
    prefix: str = "",
    default: str = "",
) -> Any:
    for alias in aliases:
        key = f"{prefix}{alias}" if prefix else alias
        if key in sample and sample[key] is not None:
            return sample[key]
    return default


def _stringify_history(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        lines = []
        for turn in value:
            if isinstance(turn, Mapping):
                speaker = turn.get("role") or turn.get("speaker") or turn.get("from")
                text = turn.get("content") or turn.get("text") or turn.get("utterance") or ""
                lines.append(f"{speaker}: {text}" if speaker else str(text))
            else:
                lines.append(str(turn))
        return "\n".join(lines)
    return str(value)


def _format_options(strategies: Sequence[str], action_tokens: Sequence[str]) -> str:
    return "\n".join(
        f"strategy #{action_token} {strategy}"
        for action_token, strategy in zip(action_tokens, strategies)
    )


def build_high_level_prompt(
    sample: Mapping[str, Any],
    strategies: Sequence[str],
    action_tokens: Sequence[str],
    *,
    prefix: str = "",
) -> str:
    emotion = _value_from_aliases(sample, ("emotion", "emo", "user_emotion"), prefix=prefix, default="")
    description = _value_from_aliases(
        sample,
        ("description", "desc", "situation", "topic"),
        prefix=prefix,
        default="",
    )
    history = _stringify_history(
        _value_from_aliases(sample, ("history", "hist", "dialogue_history"), prefix=prefix, default="")
    )
    query = _value_from_aliases(
        sample,
        ("query", "current_query", "utterance", "input"),
        prefix=prefix,
        default="",
    )
    options = _format_options(strategies, action_tokens)
    choices = " through ".join((action_tokens[0], action_tokens[-1]))
    return (
        "You are given a multi-turn dialogue between a user and an assistant. "
        "The user's basic situation is as follows:\n"
        f"Emotion: {emotion}\n"
        f"Description: {description}\n"
        "Below is the dialogue history between the user and the assistant:\n"
        f"{history}\n"
        "The user's current query is:\n"
        f"{query}\n"
        "Based on the above context, please select the most appropriate response "
        "strategy from the following options:\n"
        f"{options}\n"
        f"Please provide your selection in the format of {choices}. Your selection is:"
    )


def build_low_level_prompt(
    sample: Mapping[str, Any],
    strategy_name: str,
    *,
    prefix: str = "",
) -> str:
    emotion = _value_from_aliases(sample, ("emotion", "emo", "user_emotion"), prefix=prefix, default="")
    description = _value_from_aliases(
        sample,
        ("description", "desc", "situation", "topic"),
        prefix=prefix,
        default="",
    )
    history = _stringify_history(
        _value_from_aliases(sample, ("history", "hist", "dialogue_history"), prefix=prefix, default="")
    )
    query = _value_from_aliases(
        sample,
        ("query", "current_query", "utterance", "input"),
        prefix=prefix,
        default="",
    )
    return (
        "You are given a multi-turn dialogue between a user and an assistant. "
        "The user's basic situation is as follows:\n"
        f"Emotion: {emotion}\n"
        f"Description: {description}\n"
        "Below is the dialogue history between the user and the assistant:\n"
        f"{history}\n"
        "The user's current query is:\n"
        f"{query}\n"
        "The current response strategy is:\n"
        f"{strategy_name}\n"
        "Based on the current response strategy and other information, please act "
        "as an assistant and provide the best response. Keep replies brief without "
        "additional pronouns or extra elements."
    )


def build_evaluation_prompt(sample: Mapping[str, Any], response: str) -> str:
    emotion = _value_from_aliases(sample, ("emotion", "emo", "user_emotion"), default="")
    description = _value_from_aliases(sample, ("description", "desc", "situation", "topic"), default="")
    history = _stringify_history(_value_from_aliases(sample, ("history", "hist", "dialogue_history"), default=""))
    query = _value_from_aliases(sample, ("query", "current_query", "utterance", "input"), default="")
    return (
        "You are an expert evaluator simulating human assessment of dialogue responses. "
        "You are given a multi-turn dialogue between a user and an assistant. "
        "The user's basic situation is as follows:\n"
        f"Emotion: {emotion}\n"
        f"Description: {description}\n"
        "Below is the dialogue history between the user and the assistant:\n"
        f"{history}\n"
        "The user's current query is:\n"
        f"{query}\n"
        "The assistant's response is:\n"
        f"{response}\n"
        "Please evaluate the assistant's response across five binary-scored dimensions.\n"
        "1. Acceptance - Is the response socially appropriate and non-offensive?\n"
        "2. Effectiveness - Does the response address the user's intent appropriately?\n"
        "3. Sensitivity - Does the response consider emotional or situational context?\n"
        "4. Fluency - Is the response grammatically correct and fluent?\n"
        "5. Emotion - Does the response convey an appropriate emotional tone?\n"
        "Each dimension should be scored as 0 (unsatisfactory) or 1 (satisfactory).\n"
        "Output format:\n"
        "Acceptance: [0/1], Effectiveness: [0/1]\n"
        "Sensitivity: [0/1], Fluency: [0/1]\n"
        "Emotion: [0/1], Total score: [0-5]\n"
        "Explanation: [your reasoning here]"
    )
