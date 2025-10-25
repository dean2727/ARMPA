from datetime import datetime
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from litellm import embedding, completion
import uuid
import os
from dotenv import load_dotenv
load_dotenv()

from memory.prompts.storage import summarize_observation_prompt


class MemoryManager:
    def __init__(self, collection_name: str = "trajectory_learnings"):
        self.embed_model = "together_ai/BAAI/bge-large-en-v1.5"
        self.summarize_model = "together_ai/Qwen/Qwen3-Next-80B-A3B-Instruct" # qwen-3.5-8b-instruct

        self.client = QdrantClient(
            os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )  # replace with host="your-qdrant-url", api_key="..." if using cloud
        self.collection_name = collection_name

        # Create collection if not exists
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
            )
        except Exception as e:
            if "already exists" in str(e):
                print(f"Collection '{self.collection_name}' already exists, using existing collection.")
            else:
                raise e

    # ---------- STORE ---------- #
    def store_trajectory(self, trajectory: List[Dict[str, Any]], goal: str, success: bool):
        points = []
        step_id = 0

        # Store initial observation (no action that led to it)
        initial_state = trajectory[0]
        initial_obs_text = self._summarize_obs(initial_state["observation"]["text"])
        
        # Store initial step
        initial_point = self._store_trajectory_step(
            goal=goal,
            obs_text=initial_obs_text,
            action_taken="",  # No action for initial state
            success=success,
            step_id=step_id
        )
        points.append(initial_point)

        # Loop starting from 2nd observation (index 2), referencing action from prior step
        for i in range(2, len(trajectory), 2):  # iterate (State -> Action -> Next State)
            if i + 1 >= len(trajectory): 
                break

            current_state = trajectory[i]
            prior_action = trajectory[i - 1]  # Action that led to current state

            # --- Summarize environment cues ---
            current_obs_text = self._summarize_obs(current_state["observation"]["text"])

            # Store this step
            point = self._store_trajectory_step(
                goal=goal,
                obs_text=current_obs_text,
                action_taken=self._describe_action(prior_action),
                success=success,
                step_id=step_id
            )
            points.append(point)
            step_id += 1

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"✅ Stored {len(points)} step memories.")

    def _store_trajectory_step(self, goal: str, obs_text: str, action_taken: str, 
                              success: bool, step_id: int) -> rest.PointStruct:
        """Store a single trajectory step and return the point"""
        
        # Create cue embedding using the helper method
        cue_emb = self._create_cue_embedding(goal, action_taken, obs_text)
        
        # Create metadata
        metadata = {
            "memory_id": str(uuid.uuid4()),
            "step_id": step_id,
            "goal": goal,
            "obs_summary": obs_text,
            "action_taken": action_taken,
            "success": success,
            "strength": 0.5 if success else 0.3,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        return rest.PointStruct(id=metadata["memory_id"], vector=cue_emb, payload=metadata)

    # ---------- RETRIEVE ---------- #
    def recall(self, current_obs: Dict[str, Any], goal: str, top_k: int = 3):
        cue_emb = self._create_cue_embedding(goal, last_action, current_obs)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=cue_emb,
            limit=top_k,
        )

        recalls = []
        for r in results:
            m = r.payload
            recalls.append(
                {
                    "score": r.score,
                    "goal": m["goal"],
                    "action": m["action"],
                    "obs_summary": m["obs_summary"],
                    "result_summary": m["result_summary"],
                    "lesson": self._derive_lesson(m),
                }
            )

        return recalls


    def _create_cue_embedding(goal: str, current_obs: str, last_action: str = None):
        # A cue consists of <goal> | <what I last did> | <what I now see (summarized)>
        cue_text = f"{goal} | {current_obs} | {last_action if last_action else ''}"
        cue_emb = embedding(model=self.embed_model, input=[cue_text])
        cue_emb = cue_emb["data"][0]["embedding"]
        return cue_emb

    # ---------- UTILITIES ---------- #
    def _summarize_obs(self, obs_text: str) -> str:
        summary_prompt = f"""Summarize the following web observation according to the instructions:
        {raw_observation}
        """

        response = completion(model="together_ai/Qwen/Qwen3-Next-80B-A3B-Instruct", 
                            system=system_prompt, 
                            messages=[{"role": "user", "content": summary_prompt}],
                            temperature=0.2,
                            max_tokens=200)

        return response["choices"][0]["message"]["content"]

    # TODO
    def _derive_lesson(self, metadata: Dict[str, Any]) -> str:
        if not metadata["success"]:
            return "Avoid navigating to external links when searching internal reports."
        return "Correctly located the reporting dashboard; reuse approach."

if __name__ == "__main__":
    mm = MemoryManager(collection_name="testing")

    # Simulated failed trajectory (based on your Magento example)
    trajectory = [
        {
            "obs": {
                "url": "http://admin/dashboard",
                "visible_elements": ["DASHBOARD", "SALES", "REPORTS", "CATALOG"],
            },
            "action": "click [753]",
            "next_obs": {
                "url": "http://admin/dashboard#reports",
                "visible_elements": ["Reports", "Bestsellers", "Customers"],
            },
        },
        {
            "obs": {
                "url": "http://admin/dashboard#reports",
                "visible_elements": ["Reports", "Bestsellers", "Advanced Reporting"],
            },
            "action": "goto [http://admin/analytics/reports/show/]",
            "next_obs": {
                "url": "https://experienceleague.adobe.com/reports-menu",
                "visible_elements": ["Adobe", "Commerce", "Documentation"],
            },
        },
    ]

    # Store failed trajectory
    mm.store_trajectory(trajectory, "Find top-1 best-selling brand in Q1 2022", success=False)

    # Simulate new situation for recall
    current_obs = {
        "url": "http://admin/dashboard",
        "visible_elements": ["REPORTS", "Bestsellers", "Customers"],
    }

    recalls = mm.recall(current_obs, "Find best-selling report")
    print("\n🧠 Retrieved memories:")
    for r in recalls:
        print(f"- ({r['score']:.3f}) {r['lesson']}")