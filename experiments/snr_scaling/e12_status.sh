#!/bin/bash
# ---------------------------------------------------------------
# One-screen status of the E1.2 (CBraMod) sweep, and the collapse audit.
#
# For every e12cb_* / e12cbr_* cell dir: checkpoint count, whether
# latest.pth.tar exists, and the LAST logged training loss, read from the
# job log that announced that cell ("Cell: <EXP_ID>" is printed by
# train_e04_cell.sbatch). Then the two screens this recipe has needed before:
#
#   COLLAPSED  final loss within 3 % of ln(64) = 4.1589, InfoNCE's chance
#              value at batch 64 -- the cell never left the collapsed solution
#              (RESULTS_add_val_set.md, "Failure mode"; ~10 % of cells per seed).
#   HIGH       final loss > 1.30x the median of its same-S, same-arm siblings.
#              Loss scales with S, so the test is relative, never absolute.
#
# A clean screen means "nothing collapsed", NOT "everything trained well" --
# partial failures overlap the healthy range and only replication catches them.
#
# Usage (repo root, ON DELTA):  bash experiments/snr_scaling/e12_status.sh
# ---------------------------------------------------------------
set -uo pipefail

CKPT_ROOT="${CKPT_ROOT:-/work/hdd/bbnv/kkokate/eb_jepa/e12_cbramod_scaling}"
LN64=4.1589

echo "=== queue"
squeue -u "$USER" -o "%.10i %.28j %.9T %.10M %.6D %R" -h | grep -E "e12|ro_|null" || echo "(no e12 jobs queued)"
echo
echo "=== cells   (ckpts | latest | final loss | job log)"

# job log per cell: the newest log whose banner names the cell
declare -A LOG
for f in $(ls -t logs/e04_cell_*.out 2>/dev/null); do
    cell=$(grep -m1 -o "Cell: e12[A-Za-z0-9_]*" "$f" | cut -d' ' -f2)
    [ -n "$cell" ] && [ -z "${LOG[$cell]:-}" ] && LOG[$cell]=$f
done

TMP=$(mktemp)
for d in "$CKPT_ROOT"/e12cb_* "$CKPT_ROOT"/e12cbr_*; do
    [ -d "$d" ] || continue
    cell=$(basename "$d")
    n=$(ls -1 "$d"/epoch_*.pth.tar 2>/dev/null | wc -l)
    latest=$([ -s "$d/latest.pth.tar" ] && echo yes || echo no)
    log="${LOG[$cell]:-}"
    loss="-"
    if [ -n "$log" ]; then
        # rich console wraps "loss=X" onto the line after "[Epoch N/M]"; take the last one.
        loss=$(grep -o "loss=[0-9.]*" "$log" | tail -1 | cut -d= -f2)
        [ -z "$loss" ] && loss="-"
    fi
    printf "%-32s %3s  %-3s  %-8s %s\n" "$cell" "$n" "$latest" "$loss" "${log:-(no log yet)}"
    echo "$cell $loss" >> "$TMP"
done

echo
echo "=== collapse audit (final loss vs ln(64)=$LN64 and vs same-S siblings)"
python3 - "$TMP" "$LN64" <<'PY'
import re, statistics, sys
rows = [l.split() for l in open(sys.argv[1]) if len(l.split()) == 2 and l.split()[1] != "-"]
ln64 = float(sys.argv[2])
groups = {}
for cell, loss in rows:
    m = re.match(r"^(e12cb|e12cbr)_s(\d+)_", cell)
    if not m:
        continue
    groups.setdefault((m.group(1), int(m.group(2))), []).append((cell, float(loss)))
flagged = 0
for (arm, s), items in sorted(groups.items()):
    med = statistics.median(v for _, v in items)
    for cell, v in items:
        tags = []
        if abs(v - ln64) / ln64 < 0.03:
            tags.append("COLLAPSED")
        if med > 0 and v / med > 1.30:
            tags.append(f"HIGH x{v/med:.2f}")
        if tags:
            flagged += 1
            print(f"  {cell:32s} loss={v:.4f}  S-median={med:.4f}  {' '.join(tags)}")
print(f"  {len(rows)} cells with a final loss; {flagged} flagged")
PY
rm -f "$TMP"
