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

# Optional WebArena imports (only needed for specific methods)
try:
    from webarena.browser_env.helper_functions import get_action_description
    from webarena.agent.prompts import PromptConstructor
    WEBARENA_AVAILABLE = True
except ImportError:
    WEBARENA_AVAILABLE = False
    get_action_description = None
    PromptConstructor = None

class MemoryManager:
    def __init__(self, collection_name: str = None):
        # Use BGE-large for embeddings (1024 dims, 512 token limit)
        self.embed_model = "together_ai/BAAI/bge-large-en-v1.5"
        self.summarize_model = "together_ai/OpenAI/gpt-oss-120B"

        self.client = QdrantClient(
            os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        
        # Define the three collections
        self.collection_cues = "cues"
        self.collection_trajectory_history = "trajectory-history"
        self.collection_reasoningbank = "reasoningbank"
        
        # For backward compatibility, if collection_name is provided, use it as a prefix
        if collection_name:
            self.collection_cues = f"{collection_name}-cues"
            self.collection_trajectory_history = f"{collection_name}-trajectory-history"
            self.collection_reasoningbank = f"{collection_name}-reasoningbank"

        # Create all three collections if they don't exist
        self._ensure_collection(self.collection_cues)
        self._ensure_collection_without_vectors(self.collection_trajectory_history)
        self._ensure_collection(self.collection_reasoningbank)
        
        # Create payload indexes for filtering
        self._ensure_goal_index(self.collection_cues)
        self._ensure_goal_index(self.collection_trajectory_history)
        self._ensure_payload_index(self.collection_trajectory_history, "success", rest.PayloadSchemaType.BOOL)

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

        self.client.upsert(collection_name=self.collection_cues, points=points)
        #print(f"✅ Stored {len(points)} step memories.")
        
        # Also store trajectory history entry
        self.store_trajectory_history(goal=goal, num_steps=len(points), success=success)

    def store_trajectory_testing(self, trajectory: List[Dict[str, Any]], goal: str, success: bool, prompt_constructor = None):
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

        self.client.upsert(collection_name=self.collection_cues, points=points)
        print(f"✅ Stored {len(points)} step memories.")
        
        # Also store trajectory history entry
        self.store_trajectory_history(goal=goal, num_steps=len(points), success=success)

    def store_trajectory_history(self, goal: str, num_steps: int, success: bool, additional_metadata: Dict[str, Any] = None):
        """Store a trajectory history entry (no embeddings, just metadata)"""
        trajectory_id = str(uuid.uuid4())
        
        # Create metadata
        metadata = {
            "trajectory_id": trajectory_id,
            "goal": goal,
            "num_steps": num_steps,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        # Add any additional metadata if provided
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # No vector needed since we only query by metadata
        # Use zero vector as fallback (some Qdrant versions require vectors even for metadata-only collections)
        zero_vector = [0.0] * 1024
        point = rest.PointStruct(
            id=trajectory_id,
            vector={},
            payload=metadata
        )
        
        self.client.upsert(collection_name=self.collection_trajectory_history, points=[point])
        #print(f"✅ Stored trajectory history entry: {num_steps} steps, success={success}")

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
    def cue_based_recall(self, summarized_obs: str, goal: str, top_k: int = 3, return_embeddings: bool = True):
        """
        Retrieve top-K memories based on cue similarity.
        
        Args:
            summarized_obs: Summarized current observation
            goal: Task goal/intent
            top_k: Number of memories to retrieve
            return_embeddings: If True, include memory embeddings in output (needed for RL filter)
        
        Returns:
            recalls: List of memory dicts with metadata and optionally embeddings
        """
        cue_emb = self._create_cue_embedding(goal, summarized_obs, last_action=None)

        results = self.client.search(
            collection_name=self.collection_cues,
            query_vector=cue_emb,
            limit=top_k,
            with_vectors=return_embeddings,  # Request vectors from Qdrant if needed
        )

        recalls = []
        for r in results:
            m = r.payload
            memory_dict = {
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
            
            # Add embedding if requested (for RL filter)
            if return_embeddings and r.vector is not None:
                memory_dict["embedding"] = r.vector
            
            recalls.append(memory_dict)

        return recalls

    def get_memories_by_goal(self, goal: str, limit: int = None) -> List[Dict[str, Any]]:
        """Retrieve cue-based memories by searching the 'goal' metadata field"""
        try:
            # Use scroll with a filter to find memories matching the goal
            points, _ = self.client.scroll(
                collection_name=self.collection_cues,
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

    def get_best_trajectory_sample(self, goal: str = None, require_success: bool = True) -> Dict[str, Any] | None:
        """
        Retrieve the best trajectory sample from trajectory history.
        Best = shortest, most successful trajectory.
        
        Args:
            goal: Optional goal to filter by. If None, searches all trajectories.
            require_success: If True, prefer successful trajectories (default: True).
                           If True and no successful trajectories exist, falls back to 
                           returning the shortest trajectory (even if unsuccessful).
        
        Returns:
            Single trajectory history entry (dict) if found, None otherwise.
            If require_success=True but no successful trajectories exist, returns the shortest trajectory.
        """
        try:
            def _get_trajectories(filter_conditions):
                """Helper to get trajectories with given filter conditions"""
                if filter_conditions:
                    scroll_filter = rest.Filter(must=filter_conditions)
                else:
                    scroll_filter = None
                
                points, _ = self.client.scroll(
                    collection_name=self.collection_trajectory_history,
                    scroll_filter=scroll_filter,
                    limit=None,
                    with_payload=True,
                    with_vectors=False
                )
                
                trajectories = []
                for point in points:
                    payload = point.payload
                    trajectories.append({
                        "trajectory_id": payload.get("trajectory_id"),
                        "goal": payload.get("goal"),
                        "num_steps": payload.get("num_steps"),
                        "success": payload.get("success"),
                        "timestamp": payload.get("timestamp"),
                        **{k: v for k, v in payload.items() 
                           if k not in ["trajectory_id", "goal", "num_steps", "success", "timestamp"]}
                    })
                
                return trajectories
            
            # Build filter conditions (goal only, no success filter yet)
            filter_conditions = []
            if goal:
                filter_conditions.append(
                    rest.FieldCondition(
                        key="goal",
                        match=rest.MatchValue(value=goal)
                    )
                )
            
            # First, try to get successful trajectories if require_success=True
            if require_success:
                success_filter = filter_conditions + [
                    rest.FieldCondition(
                        key="success",
                        match=rest.MatchValue(value=True)
                    )
                ]
                trajectories = _get_trajectories(success_filter)
                
                # If we found successful trajectories, return the shortest one
                if trajectories:
                    trajectories.sort(key=lambda x: x.get("num_steps", float('inf')))
                    return trajectories[0]
                
                # No successful trajectories found, fall back to shortest overall
                # (remove success filter, keep goal filter if any)
                trajectories = _get_trajectories(filter_conditions)
                if not trajectories:
                    return None
                # Sort by num_steps (ascending) to get shortest
                trajectories.sort(key=lambda x: x.get("num_steps", float('inf')))
                return trajectories[0]
            else:
                # Not requiring success, just get all matching trajectories
                trajectories = _get_trajectories(filter_conditions)
                if not trajectories:
                    return None
                # Sort by: success first, then by num_steps (ascending)
                trajectories.sort(key=lambda x: (
                    0 if x.get("success", False) else 1,  # Successful first
                    x.get("num_steps", float('inf'))  # Then by num_steps ascending
                ))
                return trajectories[0]
            
        except Exception as e:
            print(f"❌ Error retrieving best trajectory sample: {e}")
            return None

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

    def _get_embedding(self, text: str):
        """
        Get embedding for a single text string.
        
        This is a helper method for RL filter integration, allowing
        embeddings to be generated for task goals and observations separately.
        
        Args:
            text: Input text to embed
        
        Returns:
            embedding: 1024-dim numpy array or list
        """
        emb = embedding(model=self.embed_model, input=[text])
        return emb["data"][0]["embedding"]
    
    def _create_cue_embedding(self, goal: str, current_obs: str, last_action: str = None):
        """
        Create cue embedding for memory retrieval.
        
        A cue consists of <goal> | <what I last did> | <what I now see (summarized)>
        BGE-large has 512 token limit. Empirically: ~4 chars per token.
        Strategy: Truncate each field to ensure total stays under 512 tokens (~2000 chars total)
        
        Args:
            goal: Task goal/intent
            current_obs: Current observation (summarized)
            last_action: Optional previous action taken
        
        Returns:
            cue_emb: 1024-dim embedding vector
        """
        max_goal_chars = 400
        max_obs_chars = 1200  # Observations are most important
        max_action_chars = 300
        
        goal_trunc = goal[:max_goal_chars] if len(goal) > max_goal_chars else goal
        obs_trunc = current_obs[:max_obs_chars] if len(current_obs) > max_obs_chars else current_obs
        action_trunc = last_action[:max_action_chars] if last_action and len(last_action) > max_action_chars else last_action
        
        cue_text = f"{goal_trunc} | {obs_trunc}{f' | {action_trunc}' if action_trunc else ''}"
        
        cue_emb = embedding(model=self.embed_model, input=[cue_text])
        cue_emb = cue_emb["data"][0]["embedding"]
        return cue_emb

    # ---------- INSPECT ---------- #
    def print_all_memories(self, limit: int = None, collection: str = None, pages: int = None):
        """
        Pretty print all memories, with optional page-based scroll.
        
        Args:
            limit: Number of points per page. If None, Qdrant default (64) is used.
            collection: Qdrant collection name.
            pages: Number of pages to scroll. If None, scroll until the end.
        """
        collection_name = collection or self.collection_cues

        offset = None
        page_count = 0
        total_count = 0
        all_goals = set()

        try:
            while True:
                points, offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=limit,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                if not points:
                    break

                page_count += 1

                print(f"\n📄 Page {page_count}")
                print("=" * 80)

                for point in points:
                    total_count += 1
                    payload = point.payload
                    all_goals.add(payload.get('goal', 'N/A'))
                    print(f"\n--- Memory {total_count} (ID: {point.id}) ---")
                    print(f"📝 Goal: {payload.get('goal', 'N/A')}")
                    print(f"👁️  Observation: {payload.get('obs_summary', 'N/A')}")
                    print(f"🎯 Action: {payload.get('action_taken', 'N/A')}")
                    print(f"✅ Success: {payload.get('success', 'N/A')}")
                    print(f"💪 Strength: {payload.get('strength', 'N/A')}")
                    print(f"🕒 Timestamp: {payload.get('timestamp', 'N/A')}")
                    print(f"🔢 Step ID: {payload.get('step_id', 'N/A')}")

                # If user requested only a certain number of pages
                if pages is not None and page_count >= pages:
                    break

                # If no more pages
                if offset is None:
                    break
            
            print(f"\n🎯 Unique goals (count: {len(all_goals)}): {all_goals}")
            if total_count == 0:
                print(f"📭 No memories found in the collection '{collection_name}'.")
            else:
                print(f"\n🧠 Total memories printed: {total_count}")

        except Exception as e:
            print(f"❌ Error retrieving memories: {e}")
    
    def print_memory_stats(self, collection: str = None):
        """Print statistics about the memory collection"""
        # Default to cues collection if not specified
        collection_name = collection or self.collection_cues
        try:
            # Get collection info
            collection_info = self.client.get_collection(collection_name)
            total_points = collection_info.points_count
            
            # Get a sample of points to analyze
            points, _ = self.client.scroll(
                collection_name=collection_name,
                limit=100,  # Sample first 100 points
                with_payload=True,
                with_vectors=False
            )
            
            if not points:
                print(f"📭 No memories found in the collection '{collection_name}'.")
                return
            
            # Calculate statistics
            success_count = sum(1 for p in points if p.payload.get('success', False))
            failure_count = total_points - success_count
            
            goals = [p.payload.get('goal', '') for p in points if p.payload.get('goal')]
            unique_goals = len(set(goals))
            
            strengths = [p.payload.get('strength', 0) for p in points if p.payload.get('strength')]
            avg_strength = sum(strengths) / len(strengths) if strengths else 0
            
            print(f"📊 Memory Collection Statistics:")
            print(f"   Collection: {collection_name}")
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
    def reset_collection(self, collection_name: str = None):
        """
        Reset/wipe a specific collection by deleting and recreating it.
        
        Args:
            collection_name: Name of the collection to reset. If None, resets all collections.
                            Can be one of: 'cues', 'trajectory-history', 'reasoningbank',
                            or the full collection name (e.g., 'testing-cues').
        """
        if collection_name is None:
            # Reset all collections
            self.reset_collection('cues')
            self.reset_collection('trajectory-history')
            self.reset_collection('reasoningbank')
            return
        
        # Determine which collection this is based on the name
        # Handle both short names and full names (with prefix)
        is_cues = collection_name == 'cues' or collection_name == self.collection_cues or collection_name.endswith('-cues')
        is_trajectory_history = collection_name == 'trajectory-history' or collection_name == self.collection_trajectory_history or collection_name.endswith('-trajectory-history')
        is_reasoningbank = collection_name == 'reasoningbank' or collection_name == self.collection_reasoningbank or collection_name.endswith('-reasoningbank')
        
        # Get the actual collection name to use
        if is_cues:
            actual_collection_name = self.collection_cues
            has_vectors = True
        elif is_trajectory_history:
            actual_collection_name = self.collection_trajectory_history
            has_vectors = False
        elif is_reasoningbank:
            actual_collection_name = self.collection_reasoningbank
            has_vectors = True
        else:
            # Unknown collection, try to use the provided name as-is
            actual_collection_name = collection_name
            # Default to assuming it has vectors
            has_vectors = True
            print(f"⚠️  Warning: Unknown collection '{collection_name}', assuming it has vectors")
        
        # Delete and recreate the collection
        try:
            self.client.delete_collection(collection_name=actual_collection_name)
            print(f"🗑️  Deleted collection '{actual_collection_name}'")
        except Exception as e:
            if "doesn't exist" in str(e).lower():
                print(f"ℹ️  Collection '{actual_collection_name}' doesn't exist, will create it")
            else:
                raise e
        
        # Recreate the collection
        if has_vectors:
            self._ensure_collection(actual_collection_name)
        else:
            self._ensure_collection_without_vectors(actual_collection_name)
        print(f"✅ Recreated empty collection '{actual_collection_name}'")
        
        # Recreate the appropriate indexes
        if is_cues:
            self._ensure_goal_index(actual_collection_name)
        elif is_trajectory_history:
            self._ensure_goal_index(actual_collection_name)
            self._ensure_payload_index(actual_collection_name, "success", rest.PayloadSchemaType.BOOL)
        # reasoningbank doesn't have indexes yet (TODO)

    # ---------- UTILITIES ---------- #
    def _ensure_collection(self, collection_name: str):
        """Ensure that a collection exists with vectors, create it if it doesn't"""
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
            )
        except Exception as e:
            if "already exists" in str(e):
                print(f"Collection '{collection_name}' already exists, using existing collection.")
            else:
                raise e
    
    def _ensure_collection_without_vectors(self, collection_name: str):
        """Ensure that a collection exists without vectors (metadata only), create it if it doesn't"""
        try:
            # Try creating collection without vectors_config first
            # If that fails, fall back to using zero vectors (some Qdrant versions require vectors)
            try:
                self.client.create_collection(
                    collection_name=collection_name,
                    # No vectors_config parameter = no vectors
                )
            except Exception as inner_e:
                # If creating without vectors fails, use zero vectors as fallback
                # This allows metadata-only queries while satisfying Qdrant's vector requirement
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=rest.VectorParams(size=1024, distance=rest.Distance.COSINE),
                )
                print(f"⚠️  Note: Collection '{collection_name}' created with zero vectors (Qdrant requires vectors)")
        except Exception as e:
            if "already exists" in str(e):
                print(f"Collection '{collection_name}' already exists, using existing collection.")
            else:
                raise e
    
    def _ensure_goal_index(self, collection_name: str):
        """Ensure that a payload index exists on the 'goal' field for filtering"""
        self._ensure_payload_index(collection_name, "goal", rest.PayloadSchemaType.KEYWORD)
    
    def _ensure_payload_index(self, collection_name: str, field_name: str, field_schema: rest.PayloadSchemaType):
        """Ensure that a payload index exists on a specific field for filtering"""
        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )
            print(f"✅ Created payload index on '{field_name}' field for collection '{collection_name}'")
        except Exception as e:
            if "already exists" in str(e).lower() or "index" in str(e).lower():
                # Index already exists, which is fine
                pass
            else:
                print(f"⚠️  Warning: Could not create index on '{field_name}' field: {e}")

    def summarize_webarena_observation(self, obs_text: str, model: str = "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo", temperature: float = 0.0, max_tokens: int = 1000) -> str:
        # Truncate extremely long observations to avoid API errors
        MAX_OBS_LENGTH = 8000  # chars
        if len(obs_text) > MAX_OBS_LENGTH:
            obs_text = obs_text[:MAX_OBS_LENGTH] + "\n... [truncated]"
        
        query = f"""Summarize the following web observation according to the instructions:
        {obs_text}
        """

        try:
            response = completion(model=model, 
                                system=summarize_observation_prompt, 
                                messages=[{"role": "user", "content": query}],
                                temperature=0.8,
                                max_tokens=200)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            # If summarization fails, return truncated raw observation
            print(f"⚠️  LLM summarization failed: {e}")
            print(f"   Using truncated raw observation instead")
            return obs_text[:500] + "..." if len(obs_text) > 500 else obs_text

    # TODO: ReasoningBank stuff here
    def _derive_lesson(self, metadata: Dict[str, Any]) -> str:
        if not metadata["success"]:
            return "Avoid navigating to external links when searching internal reports."
        return "Correctly located the reporting dashboard; reuse approach."

if __name__ == "__main__":
    print("=" * 80)
    print("Testing MemoryManager with trajectory-history functionality")
    print("=" * 80)
    
    mm = MemoryManager(collection_name="testing")
    
    # Uncomment the line below to reset/wipe all collections
    # mm.reset_collection()  # Reset all collections
    # Or reset a specific collection:
    # mm.reset_collection('cues')
    mm.reset_collection('testing-trajectory-history')
    # mm.reset_collection('reasoningbank')
    
    print("\n1. Testing trajectory history storage...")
    print("-" * 80)
    
    # Store some test trajectories with different goals and success rates
    test_trajectories = [
        ("Find user profile", 5, True),   # Short successful
        ("Find user profile", 8, True),   # Longer successful
        ("Find user profile", 3, False),  # Short failed
        ("Find user profile", 10, False), # Long failed
        ("Search products", 4, False),     # Short successful, different goal
        ("Search products", 7, False),    # Failed, different goal
    ]
    
    for goal, num_steps, success in test_trajectories:
        mm.store_trajectory_history(goal=goal, num_steps=num_steps, success=success)
    
    print("\n2. Testing get_best_trajectory_sample()...")
    print("-" * 80)
    
    # Test 1: Get best trajectory for "Find user profile" (should return shortest successful = 5 steps)
    print("\nTest 1: Best trajectory for 'Find user profile' (require_success=True)")
    best = mm.get_best_trajectory_sample(goal="Find user profile", require_success=True)
    if best:
        print(f"✅ Found: {best['num_steps']} steps, success={best['success']}, goal='{best['goal']}'")
    else:
        print("❌ No trajectory found")
    
    # Test 2: Get best trajectory when no successful ones exist (should fall back to shortest = 3 steps)
    print("\nTest 2: Best trajectory for 'Find user profile' with only failed trajectories")
    # First, let's manually test by creating a scenario with only failures
    mm.store_trajectory_history(goal="Only failures", num_steps=6, success=False)
    mm.store_trajectory_history(goal="Only failures", num_steps=2, success=False)  # Shortest
    mm.store_trajectory_history(goal="Only failures", num_steps=9, success=False)
    
    best = mm.get_best_trajectory_sample(goal="Only failures", require_success=True)
    if best:
        print(f"✅ Found (fallback to shortest): {best['num_steps']} steps, success={best['success']}, goal='{best['goal']}'")
        print(f"   (Should be 2 steps, even though it's unsuccessful)")
    else:
        print("❌ No trajectory found")
    
    # Test 3: Get best trajectory without requiring success
    print("\nTest 3: Best trajectory for 'Find user profile' (require_success=False)")
    best = mm.get_best_trajectory_sample(goal="Find user profile", require_success=False)
    if best:
        print(f"✅ Found: {best['num_steps']} steps, success={best['success']}, goal='{best['goal']}'")
        print(f"   (Should prefer successful, then shortest)")
    else:
        print("❌ No trajectory found")
    
    # Test 4: Get best trajectory for different goal
    print("\nTest 4: Best trajectory for 'Search products' (require_success=True)")
    best = mm.get_best_trajectory_sample(goal="Search products", require_success=True)
    if best:
        print(f"✅ Found: {best['num_steps']} steps, success={best['success']}, goal='{best['goal']}'")
    else:
        print("❌ No trajectory found")
    
    # Test 5: Get best trajectory with no goal filter
    print("\nTest 5: Best trajectory overall (no goal filter, require_success=True)")
    best = mm.get_best_trajectory_sample(goal=None, require_success=True)
    if best:
        print(f"✅ Found: {best['num_steps']} steps, success={best['success']}, goal='{best['goal']}'")
    else:
        print("❌ No trajectory found")
    
    print("\n3. Testing collection statistics...")
    print("-" * 80)
    mm.print_memory_stats(collection=mm.collection_trajectory_history)
    
    print("\n4. Testing trajectory history inspection...")
    print("-" * 80)
    mm.print_all_memories(limit=10, collection=mm.collection_trajectory_history)
    
    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)