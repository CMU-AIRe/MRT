# arm_b_variant2.sh — Arm B, variant-ii ("mixed", Fig-6 caption reading):
# terminations participate in stock-style group normalization instead of an
# explicit bonus (repair alt #1 in implementation_plan.md, assumption A6).
#
# Everything else is identical to arm_b.sh; ARM resolves to arm_b_mixed there,
# so checkpoints land in runs/arm_b_mixed/ and wandb group is arm_b_mixed-a*.

export MRT_VARIANT=mixed

_CFG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_CFG_DIR}/arm_b.sh"
