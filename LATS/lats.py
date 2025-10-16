#!/usr/bin/env python
# coding: utf-8

import os
import openai
import backoff
import sys
import copy
import itertools
import numpy as np
from functools import partial
import requests
import logging
import random

# Import the models module (assuming it's available in the parent directory)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LanguageAgentTreeSearch.webshop.models import gpt

class LATS:
    def __init__(self):
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.reflection_map = []
        self.failed_trajectories = []
    
    def softmax(self, x, temperature=1.0):
        e_x = np.exp((x - np.max(x)) / temperature)
        return e_x / e_x.sum(axis=0)

    def select_node_softmax(self, node, temperature=1.0):
        while node and node.children:
            uct_values = [child.uct() for child in node.children if not child.is_terminal]
            if not uct_values:
                return None  # All children are terminal

            probabilities = self.softmax(np.array(uct_values), temperature)
            selected_child = np.random.choice([child for child in node.children if not child.is_terminal], p=probabilities)
            
            node = selected_child
            
        return node

    def get_value(self, task, x, y, n_evaluate_sample, cache_value=True):
        #unique_trajectories = get_unique_trajectories(self.failed_trajectories)
        value_prompt = task.value_prompt_wrap(x, y, self.failed_trajectories, self.reflection_map)
        logging.info(f"Current: {x}")
        logging.info(f"Current: {y}")
        if cache_value and value_prompt in task.value_cache:
            return task.value_cache[value_prompt]
        logging.info(f"VALUE PROMPT: {value_prompt}")
        value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
        logging.info(f"VALUE OUTPUTS: {value_outputs}")
        value = task.value_outputs_unwrap(value_outputs)
        logging.info(f"VALUES: {value}")
        if cache_value:
            task.value_cache[value_prompt] = value
        return value

    def get_values(self, task, x, ys, n_evaluate_sample, cache_value=False):
        values = []
        local_value_cache = {}
        for y in ys:  # each partial output
            if y in local_value_cache:  # avoid duplicate candidates
                value = 0
            else:    
                value = self.get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
                local_value_cache[y] = value
            values.append(value)
        return values

    def get_samples(self, task, x, y, n_generate_sample, prompt_sample, stop):
        #print("MCTS FAIELD", self.failed_trajectories)
        #unique_trajectories = get_unique_trajectories(self.failed_trajectories)
        #unique_trajectories = self.failed_trajectories
        #print(len(unique_trajectories))
        #print(len(self.reflection_map))
        if len(self.failed_trajectories) > len(self.reflection_map) and len(self.failed_trajectories) < 4:
            print("generating reflections")
            print(len(self.failed_trajectories))
            print(len(self.reflection_map))
            self.reflection_map = task.generate_self_reflection(self.failed_trajectories, x)
        if prompt_sample == 'standard':
            prompt = task.standard_prompt_wrap(x, y)
        elif prompt_sample == 'cot':
            prompt = task.cot_prompt_wrap(x, y, self.reflection_map)
        else:
            raise ValueError(f'prompt_sample {prompt_sample} not recognized')
        logging.info(f"PROMPT: {prompt}")
        samples = gpt(prompt, n=n_generate_sample, stop=stop)
        return [y + _ for _ in samples]

    def get_unique_trajectories(self, failed_trajectories, num=3):
        unique_trajectories = []
        seen_final_answers = set()
        for traj in failed_trajectories:
            final_answer = traj.get('final_answer')
            if final_answer not in seen_final_answers:
                unique_trajectories.append(self.node_trajectory_to_text(traj['trajectory']))
                seen_final_answers.add(final_answer)
            if len(unique_trajectories) >= num:
                break
        return unique_trajectories

    def node_trajectory_to_text(self, node_string):
        lines = node_string.split('\n')
        formatted_lines = []
        for line in lines:
            try:
                depth = int(line.split(",")[0].split("=")[1].strip())
                action = line.split(", action=")[1].split(", observation=")[0].strip()
                observation = line.split(", observation=")[1].split(")")[0].strip()
            except IndexError:
                continue
            
            if depth != 0:
                if action:
                    formatted_lines.append(f"Action {depth}: {action}")
                if observation:
                    formatted_lines.append(f"Observation {depth}: {observation}")
        
        return '\n'.join(formatted_lines)

    def collect_actions_to_node(self, node):
        actions = []
        while node:
            if node.state['action']:
                actions.append(node.state['action'])
            node = node.parent
        return list(reversed(actions))

    def collect_all_nodes(self, node):
        """Recursively collect all nodes starting from the given node."""
        nodes = [node]
        for child in node.children:
            nodes.extend(self.collect_all_nodes(child))
        return nodes

    def collect_trajectory(self, node):
        trajectory = []
        #print("collecting traj", node)
        
        # Append the question from the root node
        trajectory.append(node.question)
        
        # Collect action and observation from each node till the root
        while node:
            if node.state and 'action' in node.state and node.state['action'] and node.parent:
                trajectory.append(f"Action: {node.state['action']}")
            else:
                logging.warning(f"Missing or empty action in node at depth {node.depth}")
                
            if node.state and 'observation' in node.state and node.state['observation'] and node.parent:
                trajectory.append(f"Observation: {node.state['observation']}\n")
            else:
                logging.warning(f"Missing or empty observation in node at depth {node.depth}")
                
            node = node.parent
        return '\n'.join(trajectory)

    def simple_search(self, args, task, idx, env, iterations=8, max_depth=15, to_print=True):
        # Initialization
        action = 'reset'
        x = env.step(idx, action)[0]
        root = Node(state=None, question=x)
        root.env_state = copy.deepcopy(env.sessions)
        successful_trajectories = []
        unsuccessful_trajectories = []
        self.failed_trajectories = []
        self.reflection_map = []

        if to_print:
            print(f"{idx}: {x}")

        # Main Loop
        for i in range(iterations):
            logging.info(f"Iteration {i + 1}")
            node = root  # Always start from the root node
            depth = 0

            # Perform a simulation from the root
            while not node.is_terminal and depth < max_depth:
                self.expand_node(node, args, task, idx, env)  # Expand current node
                if not node.children:
                    break  # If no child can be generated, break
                node = random.choice(node.children)  # Randomly select a child node
                depth += 1

            # Check the terminal condition
            if node.is_terminal and node.reward == 1:
                logging.info(f"Successful trajectory found in iteration {i + 1}")
                successful_trajectories.append(node)
                break
            elif node.is_terminal and node.reward < 1:
                logging.info(f"Unsuccessful trajectory found in iteration {i + 1}")
                unsuccessful_trajectories.append(node)

            # Reset the tree (optional)
            root.children = []

        # Post-process: select the best trajectory
        if successful_trajectories:
            best_node = max(successful_trajectories, key=lambda x: x.reward)
            return best_node.state, best_node.value, best_node.reward, best_node.em
        else:
            best_node = max(unsuccessful_trajectories, key=lambda x: x.reward)
            return best_node.state, best_node.value, best_node.reward, best_node.em

    def rollout(self, node, args, task, idx, env, max_depth=15):
        depth = 0
        n = 5
        while not node.is_terminal and depth < max_depth:
            # Generate new states
            new_states = []
            values = []
            while len(new_states) == 0:
                new_states = self.generate_new_states(node, args, task, idx, env, n)

            for state in new_states:
                if state.is_terminal:
                    return state
                    
            child_prompts = [self.generate_prompt(child) for child in new_states if not child.is_terminal and child is not None]
            #new_state = new_state[0]
            while len(values) == 0:
                values = self.get_values(task, node.question, child_prompts, args.n_evaluate_sample)
            
            max_value_index = values.index(max(values))
            node = new_states[max_value_index] 
            depth += 1
            if depth == max_depth:
                node.reward = -0.5
        return node  

    def select_node(self, node):
        while node and node.children:
            logging.info(f"Selecting from {len(node.children)} children at depth {node.depth}.")
            
            terminal_children = [child for child in node.children if child.is_terminal]
            terminal_status = [child.is_terminal for child in node.children]
            
            if len(terminal_children) == len(node.children):
                logging.info(f"All children are terminal at depth {node.depth}. Backtracking...")
                if node.parent:  
                    node.parent.children.remove(node)
                node = node.parent  
                continue  
            
            node_with_reward_1 = next((child for child in terminal_children if child.reward == 1), None)
            if node_with_reward_1:
                logging.info(f"Found terminal node with reward 1 at depth {node.depth}.")
                return node_with_reward_1
            
            node = max((child for child in node.children if not child.is_terminal), key=lambda child: child.uct(), default=None)

            while node.is_terminal and node.reward != 1:
                node = max((child for child in node.parent.children if not child.is_terminal), key=lambda child: child.uct(), default=None)
                
            logging.info(f"Selected node at depth {node.depth} with UCT {node.uct()}.")
            
        return node 

    def expand_node(self, node, args, task, idx, env):
        n = args.n_generate_sample
        if node.depth >= 15:
            logging.info("Depth limit reached")
            return
        if node.depth == 0:
            n *= 2
        new_nodes = self.generate_new_states(node, args, task, idx, env, n)
        node.children.extend(new_nodes)

    def generate_new_states(self, node, args, task, idx, env, n):
        prompt = self.generate_prompt(node)
        #print(prompt)
        sampled_actions = self.get_samples(task, prompt, "\nAction: ", n, prompt_sample=args.prompt_sample, stop="Observation")
        logging.info(f"SAMPLED ACTION: {sampled_actions}")
        unique_states = {}  # Store unique states here
        added = False
        for action in sampled_actions:
            local_sessions = copy.deepcopy(node.env_state)
            env.sessions = local_sessions
            logging.info(env.sessions)
            new_state = node.state.copy()  # Make a copy of the parent node's state
            action_line = next((line.split(":")[1].strip() for line in action.split("\n") if line.startswith("Action") and ":" in line), None)

            # Use thought and action to form a unique key
            unique_key = f"{action_line}"
            
            # if unique_key in unique_states:
            #     continue  # Skip if this state already exists

            if action_line:
                try:
                    res = env.step(idx, action_line)
                    #print("res", res)
                    obs = res[0]
                    r = res[1]
                    done = res[2]
                except AssertionError:
                    obs = 'Invalid action!'
                    # print("err")
                    r = -1
                    done = False
                
                if action.startswith('think'):
                    observation = 'OK.'
          
                # Update the new state dictionary
                new_state['action'] = action_line
                new_state['observation'] = obs
                
                env_state_clone = env.clone_state()  # Clone current environment state
                new_node = Node(state=new_state, question=node.question, env_state=env_state_clone, parent=node)
                new_node.env_state = local_sessions
                if r > 0 or done:
                    logging.info(f"reward:{r}")
                    new_node.is_terminal = True
                    #print("rew", r)
                new_node.reward = r
                new_node.value = r
                unique_states[unique_key] = new_node  # Add this state to unique_states
                logging.info(f"NEW NODE: {new_node}")

                if new_node.is_terminal and r < 1.0 and r > 0.0 and added == False:
                    trajectory = self.collect_trajectory(new_node)

                    # Check if there is already a failed trajectory with the same reward
                    existing_rewards = [t['r'] for t in self.failed_trajectories]

                    if r not in existing_rewards:
                        print("adding to failed")
                        added = True
                        self.failed_trajectories.append({'trajectory': trajectory, 'final_answer': f"{action_line}", 'r': r})

        return list(unique_states.values())  # Return unique nodes as a list

    def evaluate_node(self, node, args, task, idx):
        #actions_to_node = self.collect_actions_to_node(node)
        #env.restore_state(actions_to_node, idx)
        
        child_prompts = [self.generate_prompt(child) for child in node.children if not child.is_terminal]

        votes = self.get_values(task, node.question, child_prompts, args.n_evaluate_sample)
        
        logging.info(f"Length of votes: {len(votes)}")
        logging.info(f"Length of node.children: {len(node.children)}")
        
        # Pre-allocate votes list
        votes = votes + [0] * (len(node.children) - len(votes))
        
        max_vote = max(votes) if votes else 1
        if max_vote == 0:
            max_vote = 1  # Avoid division by zero
        
        terminal_conditions = [1 if child.is_terminal else 0 for child in node.children]
        for i, condition in enumerate(terminal_conditions):
            if condition == 1:
                votes[i] = max_vote + 1
        
        for i, child in enumerate(node.children):
            child.value = votes[i] / max_vote  # Now safe from division by zero
        
        return sum(votes) / len(votes) if votes else 0

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + value) / node.visits
            logging.info(f"Backpropagating with reward {value} at depth {node.depth}. New value: {node.value}.")
            # else:
            #     node.value = (node.value * (node.visits - 1) + value) / node.visits
            #     logging.info(f"Backpropagating at depth {node.depth}. New value: {node.value}.")

            node = node.parent

    def generate_prompt(self, node):
        trajectory = []
        question = node.question
        while node:
            new_segment = []
            if node.state['action']:
                new_segment.append(f"Action: {node.state['action']}")
            if node.state['observation'] and node.depth != 0:  # Exclude the observation from the root node
                new_segment.append(f"Observation: {node.state['observation']}")
            trajectory.append('\n'.join(new_segment))
            node = node.parent
        return question + '\n\n'.join(reversed(trajectory))

    def print_tree(self, node, level=0):
        indent = "  " * level
        print(f"{indent}{node}")
        for child in node.children:
            self.print_tree(child, level + 1)


class Node:
    def __init__(self, state, question, env_state=None, parent=None):
        self.state = {'action': '', 'observation': ''} if state is None else state
        self.parent = parent
        self.question = question
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.exhausted = False # If all children are terminal
        self.em = 0  # Exact match, evaluation metric
        self.env_state = env_state

    def uct(self):
        if self.visits == 0 and self.value >= 0:
            return float('inf')
            #return self.value * 2
        elif self.visits == 0 and self.value < 0:
            return self.value
        return self.value / self.visits + np.sqrt(2 * np.log(self.parent.visits) / self.visits)
    
    def uct_with_depth(self, C1=1, C2=1):
        if self.visits == 0:
            return self.value
        exploitation_term = self.value / self.visits
        exploration_term = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        depth_term = self.depth
        return exploitation_term + C1 * exploration_term + C2 * depth_term

    def __str__(self):
        return f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, action={self.state['action']}, observation={self.state['observation']})"
    
    def to_dict(self):
        return {
            'state': self.state,
            'question': self.question,
            'parent': self.parent.to_dict() if self.parent else None,
            'children': [child.to_dict() for child in self.children],
            'visits': self.visits,
            'value': self.value,
            'depth': self.depth,
            'is_terminal': self.is_terminal,
            'reward': self.reward,
            'em': self.em,
        }
