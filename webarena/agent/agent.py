import argparse
import json
from typing import Any

import tiktoken
from beartype import beartype

from agent.prompts import *
from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
    create_playwright_action,
)
from browser_env.utils import Observation, StateInfo
from llms import (
    call_llm,
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    generate_from_litellm_completion,
    lm_config,
)
from llms.tokenizers import Tokenizer


class Agent:
    """Base class for the agent"""

    def __init__(self, *args: Any) -> None:
        pass

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        raise NotImplementedError

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        raise NotImplementedError


class TeacherForcingAgent(Agent):
    """Agent that follows a pre-defined action sequence"""

    def __init__(self) -> None:
        super().__init__()

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def set_actions(self, action_seq: str | list[str]) -> None:
        if isinstance(action_seq, str):
            action_strs = action_seq.strip().split("\n")
        else:
            action_strs = action_seq
        action_strs = [a.strip() for a in action_strs]

        actions = []
        for a_str in action_strs:
            try:
                if self.action_set_tag == "playwright":
                    cur_action = create_playwright_action(a_str)
                elif self.action_set_tag == "id_accessibility_tree":
                    cur_action = create_id_based_action(a_str)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
            except ActionParsingError as e:
                cur_action = create_none_action()

            cur_action["raw_prediction"] = a_str
            actions.append(cur_action)

        self.actions: list[Action] = actions

    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: Any
    ) -> Action:
        """Predict the next action given the observation"""
        return self.actions.pop(0)

    def reset(
        self,
        test_config_file: str,
    ) -> None:
        with open(test_config_file) as f:
            ref_actions = json.load(f)["reference_action_sequence"]
            tag = ref_actions["action_set_tag"]
            action_seq = ref_actions["action_sequence"]
            self.set_action_set_tag(tag)
            self.set_actions(action_seq)


class PromptAgent(Agent):
    """prompt-based agent that emits action given the history"""

    @beartype
    def __init__(
        self,
        action_set_tag: str,
        lm_config: lm_config.LMConfig = None,
        prompt_constructor: PromptConstructor = None,
        instruction_path: str = None,
        model: str = "together_ai/Qwen/Qwen3-Next-80B-A3B-Instruct",
        temperature: float = 0.0,
        use_litellm: bool = False,
    ) -> None:
        super().__init__()
        self.lm_config = lm_config
        self.prompt_constructor = prompt_constructor
        self.action_set_tag = action_set_tag
        self.instruction_path = instruction_path
        self.model = model
        self.temperature = temperature
        self.use_litellm = use_litellm
        
        self.prompt_template = self._load_prompt_template()

    def set_action_set_tag(self, tag: str) -> None:
        self.action_set_tag = tag

    def _load_prompt_template(self):
        """Load prompt template from Python file like in test_llm.py"""
        with open(self.instruction_path, 'r') as f:
            content = f.read()
            # Extract the prompt dictionary from the Python file
            local_namespace = {}
            exec(content, {}, local_namespace)
            return local_namespace['prompt']

    def _construct_prompt(self, observation, url, objective, previous_action):
        """Construct the prompt using the template"""
        template = self.prompt_template["template"]
        return template.format(
            observation=observation,
            url=url,
            objective=objective,
            previous_action=previous_action
        )

    def _extract_action(self, response):
        """Extract action from LLM response using backticks"""
        import re
        action_splitter = "```"
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, response)
        if match:
            return match.group(1).strip()
        else:
            raise ValueError(f"Cannot parse action from response: {response}")

    @beartype
    def next_action(
        self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]
    ) -> Action:
        if self.use_litellm:
            # Use litellm with Python prompt files
            return self._next_action_litellm(trajectory, intent, meta_data)
        else:
            # Use original prompt constructor method
            return self._next_action_original(trajectory, intent, meta_data)

    def _next_action_litellm(self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]) -> Action:
        """Litellm-based action generation using Python prompt files"""
        # Get the latest observation
        latest_state = trajectory[-1]
        observation = latest_state["observation"]["text"]
        url = latest_state["info"]["page"].url
        objective = intent
        previous_action = meta_data.get("action_history", ["None"])[-1]
        
        # Construct the full prompt using the template structure
        full_prompt = self.prompt_template["intro"] + "\n\n"
        
        # Add examples
        if self.prompt_template["examples"]:
            full_prompt += "Here are a few examples:\n"
            for example in self.prompt_template["examples"]:
                full_prompt += f"{example[0]}\n"
                full_prompt += f"Action: {example[1]}\n\n"
        
        # Add the current observation template
        prompt = self._construct_prompt(observation, url, objective, previous_action)
        full_prompt += "Now make prediction given the observation:\n\n"
        full_prompt += prompt + "\n\n"
        full_prompt += "Action:"

        # Generate action using litellm
        response = generate_from_litellm_completion(
            prompt=full_prompt,
            model=self.model,
            temperature=self.temperature,
        )
        
        # Extract the action
        action_str = self._extract_action(response)

        print(f"Action from the LLM: {action_str}")
        
        # Create the action
        if self.action_set_tag == "id_accessibility_tree":
            action = create_id_based_action(action_str)
        elif self.action_set_tag == "playwright":
            action = create_playwright_action(action_str)
        else:
            raise ValueError(f"Unknown action type {self.action_set_tag}")
        
        action["raw_prediction"] = response
        return action

    def _next_action_original(self, trajectory: Trajectory, intent: str, meta_data: dict[str, Any]) -> Action:
        """Original prompt constructor method"""
        prompt = self.prompt_constructor.construct(
            trajectory, intent, meta_data
        )
        lm_config = self.lm_config
        n = 0
        while True:
            response = call_llm(lm_config, prompt)
            force_prefix = self.prompt_constructor.instruction[
                "meta_data"
            ].get("force_prefix", "")
            response = f"{force_prefix}{response}"
            n += 1
            try:
                parsed_response = self.prompt_constructor.extract_action(
                    response
                )
                if self.action_set_tag == "id_accessibility_tree":
                    action = create_id_based_action(parsed_response)
                elif self.action_set_tag == "playwright":
                    action = create_playwright_action(parsed_response)
                else:
                    raise ValueError(
                        f"Unknown action type {self.action_set_tag}"
                    )
                action["raw_prediction"] = response
                break
            except ActionParsingError as e:
                if n >= lm_config.gen_config["max_retry"]:
                    action = create_none_action()
                    action["raw_prediction"] = response
                    break

        return action

    def reset(self, test_config_file: str) -> None:
        pass


def construct_agent(args: argparse.Namespace) -> Agent:
    agent: Agent
    if args.agent_type == "teacher_forcing":
        agent = TeacherForcingAgent()
    elif args.agent_type == "prompt":
        llm_config = lm_config.construct_llm_config(args)
        with open(args.instruction_path) as f:
            constructor_type = json.load(f)["meta_data"]["prompt_constructor"]
        tokenizer = Tokenizer(args.provider, args.model)
        prompt_constructor = eval(constructor_type)(
            args.instruction_path, lm_config=llm_config, tokenizer=tokenizer
        )
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            lm_config=llm_config,
            prompt_constructor=prompt_constructor,
        )
    elif args.agent_type == "litellm":
        # Use PromptAgent with litellm for Python prompt files
        model = getattr(args, 'model', 'together_ai/Qwen/Qwen3-Next-80B-A3B-Instruct')
        temperature = getattr(args, 'temperature', 0.0)
        agent = PromptAgent(
            action_set_tag=args.action_set_tag,
            instruction_path=args.instruction_path,
            model=model,
            temperature=temperature,
            use_litellm=True,
        )
    else:
        raise NotImplementedError(
            f"agent type {args.agent_type} not implemented"
        )
    return agent
