# EC2 Infrastructure Issue

## Problem

The WebArena EC2 instance at `ec2-18-224-1-226.us-east-2.compute.amazonaws.com` has a **server-side configuration issue** with the Shopping Admin service (Magento).

### Symptoms

When accessing the shopping admin endpoint:
```bash
curl -I http://ec2-18-224-1-226.us-east-2.compute.amazonaws.com:7780/admin
```

The server returns:
```
HTTP/1.1 302 Found
Location: http://ec2-3-128-25-196.us-east-2.compute.amazonaws.com:7780/admin
```

This redirects to a **dead EC2 instance** (ec2-3-128-25-196), which causes:
- Playwright timeouts during authentication
- Cannot generate `shopping_admin_state.json` auth file
- Tasks requiring shopping_admin fail immediately

### Root Cause

Magento has the old EC2 base URL hardcoded in its database configuration. This is a common issue when migrating Magento instances.

## Solution Required

Your colleague needs to update the Magento base URLs in the EC2 database:

```sql
-- Connect to Magento's MySQL database on ec2-18-224-1-226
USE magento;

UPDATE core_config_data 
SET value = 'http://ec2-18-224-1-226.us-east-2.compute.amazonaws.com:7780/' 
WHERE path LIKE '%base_url%';

-- Clear Magento cache
```

Or via Magento CLI:
```bash
php bin/magento config:set web/unsecure/base_url http://ec2-18-224-1-226.us-east-2.compute.amazonaws.com:7780/
php bin/magento config:set web/secure/base_url http://ec2-18-224-1-226.us-east-2.compute.amazonaws.com:7780/
php bin/magento cache:flush
```

## Current Status

### Working Services ✓
- MAP (port 3000): HTTP 200
- SHOPPING (port 7770): HTTP 302 (redirect OK)
- GITLAB (port 8023): HTTP 302 (redirect OK)  
- REDDIT (port 9999): HTTP 200

### Broken Services ✗
- SHOPPING_ADMIN (port 7780): Redirects to dead EC2

### Generated Auth Files
- ✓ `gitlab_state.json`
- ✓ `reddit_state.json`
- ✓ `gitlab.reddit_state.json`
- ✗ `shopping_state.json` (missing)
- ✗ `shopping_admin_state.json` (missing - BLOCKED)

## Workaround

Until the EC2 is fixed, you can:

1. **Test with non-shopping-admin tasks** (tasks 7-100+ exclude shopping_admin)
2. **Use the RL filter on tasks that work** (map, reddit, gitlab, regular shopping)

Example:
```bash
cd webarena
python run.py --test_start_idx 7 --test_end_idx 20 \
  --get_memory --store_memory --collect_rl_data \
  --num_memories 10 --max_steps 15
```

Tasks 7-100 include plenty of MAP, REDDIT, GITLAB tasks.

## What Was Cleaned Up

Successfully removed all webshop experiment bloat:
- Deleted `qdrant_storage/` (local vector DB from webshop)
- Deleted `runs/webshop/` (webshop experiment results)
- Deleted `run_webshop.py` (webshop runner script)
- Deleted `webshop_env/` (webshop environment files)
- Deleted `webarena/runs/202510*` and `202511*` (failed test runs)
- Deleted `PROMPT_FIX_SUMMARY.md` and `SUMMARY.md` (redundant docs)
- Reverted all `webarena/config_files/*.json` to commit ebdf055 state
- Updated all files with new EC2 address (ec2-18-224-1-226)

## What Was Kept (RL Filter Work)

Essential RL filter agent files preserved:
- ✓ `memory/rl_filter_agent.py` (core RL policy)
- ✓ `train_rl_filter.py` (PPO training script)
- ✓ `bc_pretrain.py` (behavioral cloning pre-training)
- ✓ `validate_rl_filter.py` (validation script)
- ✓ `collect_rl_training_data.py` (data collection wrapper)
- ✓ `RL_FILTER_README.md` (usage guide)
- ✓ `RL_IMPLEMENTATION_PLAN.md` (design doc)
- ✓ `memory/manager.py` (modified for RL integration)
- ✓ `webarena/run.py` (modified for RL data collection)
- ✓ `pyproject.toml` and `uv.lock` (dependency updates)

## Next Steps

1. **Ask your colleague to fix the Magento base URL** on ec2-18-224-1-226
2. **Verify the fix**: `curl -I http://ec2-18-224-1-226.us-east-2.compute.amazonaws.com:7780/admin` should NOT redirect
3. **Regenerate auth files**: `cd webarena && bash prepare.sh`
4. **Test with shopping_admin tasks**: Tasks 0-6 require shopping_admin
5. **Collect RL training data**: Full 50-100 task run once EC2 is fixed

## Testing Connectivity

```bash
# Test all ports
for port in 3000 7770 7780 8023 9999; do 
  echo -n "Port $port: "
  curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 \
    "http://ec2-18-224-1-226.us-east-2.compute.amazonaws.com:$port" && echo " ✓" || echo " ✗"
done

# Check for redirect issues
curl -I http://ec2-18-224-1-226.us-east-2.compute.amazonaws.com:7780/admin | grep Location
# Should return nothing (no Location header) once fixed
```
