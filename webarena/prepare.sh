#!/bin/bash

# prepare the evaluation
# re-validate login information
mkdir -p ./.auth

# Try to run the auto-login script, and if cookies are expired, renew them
python -m browser_env.auto_login || python -m browser_env.auto_login --site_list all
