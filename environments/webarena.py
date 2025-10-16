from base_env import BaseEnv
import requests
import os

class WebArenaEnv(BaseEnv):
    def __init__(self):
        self.base_url = os.getenv("WEB_ARENA_URL", "http://127.0.0.1:3000")
    def reset(self, session_id):
        resp = requests.post(f"{self.base_url}/api/reset", json={"session": session_id})
        return resp.json()["observation"]
    def step(self, session_id, action):
        resp = requests.post(f"{self.base_url}/api/step", json={"session": session_id, "action": action})
        data = resp.json()
        return data["observation"], data["reward"], data["done"]


# main, env lad, then 
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    env = WebArenaEnv()
    obs = env.reset("test1")
    print(obs)
    print(env.step("test1", "navigate[https://reddit.com]"))
