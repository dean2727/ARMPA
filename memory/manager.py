from datetime import datetime
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from litellm import embedding, completion
import uuid
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()

from memory.prompts.storage import summarize_observation_prompt
from webarena.browser_env.helper_functions import get_action_description
from webarena.agent.prompts import PromptConstructor

class MemoryManager:
    def __init__(self, collection_name: str = "test"):
        self.embed_model = "together_ai/BAAI/bge-large-en-v1.5"
        self.summarize_model = "together_ai/OpenAI/gpt-oss-120B"

        self.client = QdrantClient(
            os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
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
        
        # Create payload index on "goal" field for filtering
        self._ensure_goal_index()

    # ---------- STORE ---------- #
    def store_trajectory(self, observations_actions_reasonings: List[Tuple[str, str, str]], goal: str, success: bool):
        points = []
        step_id = 0

        # Iterate over (observation -> next action) pairs
        for i in tqdm(range(0, len(observations_actions_reasonings))):
            observation_summary, action_taken, reason_for_action = observations_actions_reasonings[i]

            # Store this (cue -> next action) mapping
            point = self._get_trajectory_step_point(
                goal=goal,
                obs_text=observation_summary,
                action_taken=action_taken,
                reason_for_action=reason_for_action,
                success=success,
                step_id=step_id,
            )
            points.append(point)
            step_id += 1

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"✅ Stored {len(points)} step memories.")

    def store_trajectory_testing(self, trajectory: List[Dict[str, Any]], goal: str, success: bool, prompt_constructor: PromptConstructor = None):
        points = []
        observations_actions_reasonings = []
        step_id = 0

        # Iterate over (observation -> next action) pairs
        for i in tqdm(range(0, len(trajectory) - 1, 2)):
            observation_item = trajectory[i]
            next_action_item = trajectory[i + 1]

            # Summarize the current observation (cue)
            summarized_obs = self.summarize_webarena_observation(observation_item["observation"]["text"])

            # Describe the next action taken after this observation
            next_action_text = get_action_description(
                next_action_item, observation_item["info"]["observation_metadata"],
                action_set_tag="id_accessibility_tree",
                prompt_constructor=prompt_constructor
            )

            observations_actions_reasonings.append((summarized_obs, next_action_text, next_action_item.get('llm_reasoning')))

            # Store this (cue -> next action) mapping
            point = self._get_trajectory_step_point(
                goal=goal,
                obs_text=summarized_obs,
                action_taken=next_action_text,
                reason_for_action=next_action_item.get('llm_reasoning'),
                success=success,
                step_id=step_id,
            )
            points.append(point)
            step_id += 1

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"✅ Stored {len(points)} step memories.")

    def _get_trajectory_step_point(self, goal: str, obs_text: str, action_taken: str, 
                              reason_for_action: str, success: bool, step_id: int) -> rest.PointStruct:
        """Get a single trajectory step and return the point"""
        
        # Create cue embedding using the helper method
        cue_emb = self._create_cue_embedding(goal, obs_text)
        
        # Create metadata
        metadata = {
            "memory_id": str(uuid.uuid4()),
            "step_id": step_id,
            "goal": goal,
            "obs_summary": obs_text,
            "action_taken": action_taken,
            "reason_for_action": reason_for_action,
            "success": success,
            "strength": 0.5 if success else 0.3,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        return rest.PointStruct(id=metadata["memory_id"], vector=cue_emb, payload=metadata)

    # ---------- RETRIEVE ---------- #
    def cue_based_recall(self, summarized_obs: str, goal: str, top_k: int = 3):
        cue_emb = self._create_cue_embedding(goal, summarized_obs, last_action=None)

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
                    "score": r.score,  # Score from the search result
                    "memory_id": m["memory_id"],
                    "step_id": m["step_id"],
                    "goal": m["goal"],
                    "obs_summary": m["obs_summary"],
                    "action_taken": m["action_taken"],  # Corrected key from "action"
                    "success": m["success"],
                    "strength": m["strength"],
                    "timestamp": m["timestamp"],
                }
            )

        return recalls

    def get_memories_by_goal(self, goal: str, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve cue-based memories by searching the 'goal' metadata field"""
        try:
            # Use scroll with a filter to find memories matching the goal
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(
                            key="goal",
                            match=rest.MatchValue(value=goal)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            recalls = []
            for point in points:
                payload = point.payload
                recalls.append(
                    {
                        "memory_id": payload.get("memory_id"),
                        "step_id": payload.get("step_id"),
                        "goal": payload.get("goal"),
                        "obs_summary": payload.get("obs_summary"),
                        "action_taken": payload.get("action_taken"),
                        "reason_for_action": payload.get("reason_for_action"),
                        "success": payload.get("success"),
                        "strength": payload.get("strength"),
                        "timestamp": payload.get("timestamp"),
                    }
                )
            
            # Sort by step_id before returning
            recalls.sort(key=lambda x: x.get("step_id", 0))
            
            return recalls
            
        except Exception as e:
            print(f"❌ Error retrieving memories by goal: {e}")
            return []

    def get_formatted_memories_for_prompt(self, mems: List[Dict[str, Any]]):
        formatted_mems = []
        for m in mems:
            # If there is an 'obs_summary' field, then we know it's a cue-action mapping
            if 'obs_summary' in m:
                success = "success" if m['success'] else "failure"
                pointer = "(DONT DO AGAIN)" if m['success'] else ""
                formatted_mems.append(f"""Last time I was in a similar situation, I tried doing the corresponding action, and it ultimately led to {success}:

        WHAT I SAW:
        {m['obs_summary']}

        WHAT I DID{pointer}:
        {m['action_taken']}
        """)
            else: # Learned skills -> just take the embedding (TODO)
                ...
        
        return formatted_mems

    def _create_cue_embedding(self, goal: str, current_obs: str, last_action: str = None):
        # A cue consists of <goal> | <what I last did> | <what I now see (summarized)>
        cue_text = f"{goal} | {current_obs}{f' | {last_action}' if last_action else ''}"
        cue_emb = embedding(model=self.embed_model, input=[cue_text])
        cue_emb = cue_emb["data"][0]["embedding"]
        return cue_emb

    # ---------- INSPECT ---------- #
    def print_all_memories(self, limit: int = None):
        """Pretty print all memories in the collection"""
        try:
            # Get all points from the collection
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                print("📭 No memories found in the collection.")
                return
            
            print(f"🧠 Found {len(points)} memories in collection '{self.collection_name}':")
            print("=" * 80)
            
            for i, point in enumerate(points, 1):
                payload = point.payload
                print(f"\n--- Memory {i} (ID: {point.id}) ---")
                print(f"📝 Goal: {payload.get('goal', 'N/A')}")
                print(f"👁️  Observation: {payload.get('obs_summary', 'N/A')}")
                print(f"🎯 Action: {payload.get('action_taken', 'N/A')}")
                print(f"✅ Success: {payload.get('success', 'N/A')}")
                print(f"💪 Strength: {payload.get('strength', 'N/A')}")
                print(f"🕒 Timestamp: {payload.get('timestamp', 'N/A')}")
                print(f"🔢 Step ID: {payload.get('step_id', 'N/A')}")
                
        except Exception as e:
            print(f"❌ Error retrieving memories: {e}")
    
    def print_memory_stats(self):
        """Print statistics about the memory collection"""
        try:
            # Get collection info
            collection_info = self.client.get_collection(self.collection_name)
            total_points = collection_info.points_count
            
            # Get a sample of points to analyze
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,  # Sample first 100 points
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                print("📭 No memories found in the collection.")
                return
            
            # Calculate statistics
            success_count = sum(1 for p in points if p.payload.get('success', False))
            failure_count = total_points - success_count
            
            goals = [p.payload.get('goal', '') for p in points if p.payload.get('goal')]
            unique_goals = len(set(goals))
            
            strengths = [p.payload.get('strength', 0) for p in points if p.payload.get('strength')]
            avg_strength = sum(strengths) / len(strengths) if strengths else 0
            
            print(f"📊 Memory Collection Statistics:")
            print(f"   Collection: {self.collection_name}")
            print(f"   Total memories: {total_points}")
            print(f"   Successful trajectories: {success_count}")
            print(f"   Failed trajectories: {failure_count}")
            print(f"   Unique goals: {unique_goals}")
            print(f"   Average strength: {avg_strength:.3f}")
            
            if goals:
                print(f"\n🎯 Recent goals:")
                for goal in goals[:5]:  # Show first 5 goals
                    print(f"   • {goal}")
                if len(goals) > 5:
                    print(f"   ... and {len(goals) - 5} more")
                    
        except Exception as e:
            print(f"❌ Error retrieving memory stats: {e}")

    # ---------- RESET ---------- #
    def reset_database(self):
        """Reset/wipe the entire Qdrant database by deleting and recreating the collection"""
        try:
            # Delete the existing collection
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"🗑️  Deleted collection '{self.collection_name}'")
            
            # Recreate the collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
            )
            print(f"✅ Recreated empty collection '{self.collection_name}'")
            # Recreate the index
            self._ensure_goal_index()
            
        except Exception as e:
            if "doesn't exist" in str(e).lower():
                print(f"Collection '{self.collection_name}' doesn't exist, creating new one...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
                )
                print(f"✅ Created new collection '{self.collection_name}'")
                # Create the index
                self._ensure_goal_index()
            else:
                raise e

    # ---------- UTILITIES ---------- #
    def _ensure_goal_index(self):
        """Ensure that a payload index exists on the 'goal' field for filtering"""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="goal",
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )
            print(f"✅ Created payload index on 'goal' field for collection '{self.collection_name}'")
        except Exception as e:
            if "already exists" in str(e).lower() or "index" in str(e).lower():
                # Index already exists, which is fine
                pass
            else:
                print(f"⚠️  Warning: Could not create index on 'goal' field: {e}")

    def summarize_webarena_observation(self, obs_text: str) -> str:
        query = f"""Summarize the following web observation according to the instructions:
        {obs_text}
        """

        response = completion(model="together_ai/OpenAI/gpt-oss-120B", 
                            system=summarize_observation_prompt, 
                            messages=[{"role": "user", "content": query}],
                            temperature=0.8,
                            max_tokens=200)

        return response["choices"][0]["message"]["content"]

    # TODO: ReasoningBank stuff here
    def _derive_lesson(self, metadata: Dict[str, Any]) -> str:
        if not metadata["success"]:
            return "Avoid navigating to external links when searching internal reports."
        return "Correctly located the reporting dashboard; reuse approach."

if __name__ == "__main__":
    mm = MemoryManager(collection_name="testing")
    
    # Uncomment the line below to reset/wipe the database
    # mm.reset_database()
    
    # Print memory statistics
    mm.print_memory_stats()
    
    # Print all memories (limit to first 10 for demo)
    mm.print_all_memories(limit=10)

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