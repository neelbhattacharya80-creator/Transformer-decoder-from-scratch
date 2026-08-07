#!/usr/bin/env bash

# Usage: ./mean.sh train.log [steps_per_epoch]
# Default: 500 steps/epoch

LOGFILE="${1:?Usage: $0 <train.log> [steps_per_epoch]}"
STEPS_PER_EPOCH="${2:-500}"

awk -v steps="$STEPS_PER_EPOCH" '
/Batch/ {
    epoch = int($2 / steps)

    loss = $5
    ppl = $8

    gsub(/\|/, "", loss)
    gsub(/\|/, "", ppl)

    sum_loss[epoch] += loss
    sum_ppl[epoch] += ppl
    count[epoch]++
}
END {
    printf "%-6s %-12s %-16s\n", "Epoch", "Mean Loss", "Mean Perplexity"
    printf "%-6s %-12s %-16s\n", "-----", "---------", "---------------"

    for (e = 0; e in sum_loss; e++)
        printf "%-6d %-12.4f %-16.4f\n",
               e,
               sum_loss[e] / count[e],
               sum_ppl[e] / count[e]
}' "$LOGFILE"
