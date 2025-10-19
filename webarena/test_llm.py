import random
import os
import re
from browser_env import (
    Action,
    ActionTypes,
    ObservationMetadata,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    action2str,
    create_id_based_action,
    create_stop_action,
)
from llms import generate_from_litellm_completion
import json

# Load the simplest prompt template (direct, no N/A)
def load_prompt_template():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(script_dir, "agent/prompts/raw/p_direct_id_actree_2s_no_na.py")
    with open(prompt_path, 'r') as f:
        content = f.read()
        # Extract the prompt dictionary from the Python file
        # Create a local namespace to execute the code
        local_namespace = {}
        exec(content, {}, local_namespace)
        return local_namespace['prompt']

def construct_prompt(observation, url, objective, previous_action, prompt_template):
    """Construct the prompt using the template"""
    template = prompt_template["template"]
    return template.format(
        observation=observation,
        url=url,
        objective=objective,
        previous_action=previous_action
    )

def extract_action(response):
    """Extract action from LLM response using backticks"""
    action_splitter = "```"
    pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
    match = re.search(pattern, response)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError(f"Cannot parse action from response: {response}")

def login_to_admin(env, obs, info):
    """Login to the admin panel using the correct credentials and element IDs."""
    print("Initial page observation:")
    print(obs["text"][:1000])  # Show first 1000 chars of the page
    print(f"Current URL: {info['page'].url}")
    
    # Look for input elements in the observation to find the correct IDs
    # The observation contains element IDs that we can use
    observation_text = obs["text"]
    
    # Find username input field - look for input elements with name containing "user" or "login"
    username_id = None
    password_id = None
    submit_id = None
    
    # Parse the observation to find element IDs
    lines = observation_text.split('\n')
    for line in lines:
        # Look for textbox elements with specific labels
        if 'textbox' in line.lower() and 'username' in line.lower():
            # Extract element ID from the line (usually in format [ID=X])
            match = re.search(r'\[(\d+)\]', line)
            if match:
                username_id = match.group(1)
                print(f"Found username textbox with ID: {username_id}")
        elif 'textbox' in line.lower() and 'password' in line.lower():
            match = re.search(r'\[(\d+)\]', line)
            if match:
                password_id = match.group(1)
                print(f"Found password textbox with ID: {password_id}")
        elif 'button' in line.lower() and 'sign in' in line.lower():
            match = re.search(r'\[(\d+)\]', line)
            if match:
                submit_id = match.group(1)
                print(f"Found sign in button with ID: {submit_id}")
    
    # If we couldn't find specific IDs, use the correct IDs from the page observation
    if not username_id:
        username_id = "15"  # From the page observation: [15] textbox 'Username *'
        print("Using known username ID: 15")
    if not password_id:
        password_id = "17"  # From the page observation: [17] textbox 'Password *'
        print("Using known password ID: 17")
    if not submit_id:
        submit_id = "66"  # From the page observation: [66] button 'Sign in'
        print("Using known submit ID: 66")
    
    # Step 1: Fill in the username field
    username_action = create_id_based_action(f"type [{username_id}] [admin] [0]")
    obs, reward, done, truncated, info = env.step(username_action)
    print(f"Username action result - Done: {done}, Truncated: {truncated}")

    # Step 2: Fill in the password field  
    password_action = create_id_based_action(f"type [{password_id}] [admin1234] [0]")
    obs, reward, done, truncated, info = env.step(password_action)
    print(f"Password action result - Done: {done}, Truncated: {truncated}")

    # Step 3: Click the sign in button
    signin_action = create_id_based_action(f"click [{submit_id}]")
    obs, reward, done, truncated, info = env.step(signin_action)
    print(f"Signin action result - Done: {done}, Truncated: {truncated}")
    
    # Print current observation to see if login was successful
    print(f"Current URL after login: {info['page'].url}")
    print(f"Current observation: {obs['text'][:500]}...")
    
    return obs, info

def main():
    # Check for API key
    if not os.getenv("TOGETHER_API_KEY"):
        print("Please set TOGETHER_API_KEY environment variable")
        return
    
    # Load prompt template
    prompt_template = load_prompt_template()
    
    # Initialize the environment
    env = ScriptBrowserEnv(
        headless=False,
        observation_type="accessibility_tree",
        current_viewport_only=True,
        viewport_size={"width": 1280, "height": 720},
    )
    
    # Prepare the environment for a configuration defined in a json file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, "config_files/1.json")
    obs, info = env.reset(options={"config_file": config_file})

    trajectory: Trajectory = []

    # Login to the admin panel
    #obs, info = login_to_admin(env, obs, info)
    
    # Get the text observation after login
    observation = obs["text"]
    url = info["page"].url
    objective = info["goal"]  # The task objective
    previous_action = "None"
    
    print(f"Objective: {objective}")
    print(f"URL: {url}")
    print(f"Observation: {observation[:200]}...")  # Show first 200 chars
    
    # Construct the prompt
    prompt = construct_prompt(observation, url, objective, previous_action, prompt_template)
    
    # Add the intro and examples to the prompt
    full_prompt = prompt_template["intro"] + "\n\n"
    full_prompt += "Here are a few examples:\n"
    for example in prompt_template["examples"]:
        full_prompt += f"{example[0]}\n"
        full_prompt += f"Action: {example[1]}\n\n"
    full_prompt += "Now make prediction given the observation:\n\n"
    full_prompt += prompt + "\n\n"
    full_prompt += "Action:"
    
    print("\nSending prompt to LLM...")
    
    try:
        # Generate action using litellm
        response = generate_from_litellm_completion(
            prompt=full_prompt,
            model="together_ai/Qwen/Qwen3-Next-80B-A3B-Instruct",  # Using a simple model
            temperature=0.0,
        )
        
        print(f"LLM Response: {response}")
        
        # Extract the action
        action_str = extract_action(response)
        print(f"Extracted Action: {action_str}")
        
        # Create the action
        action = create_id_based_action(action_str)
        
        # Take the action
        obs, _, terminated, _, info = env.step(action)
        
        print(f"Action taken: {action_str}")
        print(f"Terminated: {terminated}")
        
    except Exception as e:
        print(f"Error: {e}")
        # Fallback to random action like the original test.py
        id = random.randint(0, 1000)
        action = create_id_based_action(f"click [{id}]")
        obs, _, terminated, _, info = env.step(action)
        print(f"Fallback action taken: click [{id}]")

if __name__ == "__main__":
    main()
