per_step_llm_judge_prompt = """You are responsible for analyzing the performance of a web navigation agent at each step of a task.
You will be the overall goal/task, a sequence of environment observations and reasons for next actions taken by the agent,
and whether the goal could be fulfilled.

For each reasoning step, score how correct or useful it was toward achieving the goal, within [0, 1] range.

TASK GOAL: \"{task}\"

FINAL SUCCESS: {success}

OBSERVATIONS AND REASONS FOR NEXT ACTIONS:
{observations_actions_reasonings}

*END TRAJECTORY*

Return your analysis in **strict JSON** format as follows:

{{
  "step_scores": [
    {{
      "step": <int>,
      "reasoning": "<brief summary of reasoning>",
      "score": <float between 0 and 1>
    }}
  ],
  "overall_comments": "<optional brief note on performance patterns>"
}}
"""