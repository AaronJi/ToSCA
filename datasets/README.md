# Datasets

ToSCA accepts Hugging Face datasets or local JSON/JSONL/CSV files through the OpenRLHF data loader.

High-level Q_H training expects utterance-level transition fields such as:

```json
{
  "desc": "<SESSION_DESCRIPTION>",
  "emo": "<USER_EMOTION>",
  "history": "<DIALOGUE_HISTORY_BEFORE_QUERY>",
  "query": "<CURRENT_USER_QUERY>",
  "strategy": "<STRATEGY_NAME_OR_INDEX>",
  "reward": 5.0,
  "next_history": "<NEXT_DIALOGUE_HISTORY>",
  "next_query": "<NEXT_USER_QUERY>",
  "done": false
}
```

Low-level PPO prompt rendering expects dialogue context plus a selected strategy:

```json
{
  "desc": "<SESSION_DESCRIPTION>",
  "emo": "<USER_EMOTION>",
  "history": "<DIALOGUE_HISTORY_BEFORE_QUERY>",
  "query": "<CURRENT_USER_QUERY>",
  "strategy": "<SELECTED_STRATEGY>"
}
```

Use the command-line `*_key` arguments when your field names differ.

