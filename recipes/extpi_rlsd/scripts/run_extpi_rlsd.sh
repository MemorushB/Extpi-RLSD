#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_env.sh"

cat >&2 <<'MSG'
ExtPI-RLSD worker-side loss support is implemented behind:
  actor_rollout_ref.actor.policy_loss.rlsd_enabled=True
and requires teacher_pi_log_probs in the actor update batch.

The next integration hook is the inline scorer that materializes teacher_pi_log_probs
after rollout and before actor update. Use the unit tests to validate the objective
until that hook is enabled.
MSG
exit 2
