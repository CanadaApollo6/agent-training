set -euo pipefail
W=/tmp/nanbeige-review-a3c96094/sync
OUT=$W/receipts
WORK=$W/work
OUT_BUCKET=stub://bucket
mkdir -p "$OUT" "$WORK"
rm() { :; }
mock_hf() { if [[ "$3" == "$WORK" ]]; then echo WORK_FAILED; return 17; fi; echo RECEIPTS_UPLOADED; }
HF=(mock_hf)
sync_out() {
    rm -rf "$W/snapshot"
    cp -a "$OUT" "$W/snapshot"
    "${HF[@]}" buckets sync "$WORK" "$OUT_BUCKET/work"
    "${HF[@]}" buckets sync "$W/snapshot" "$OUT_BUCKET/receipts"
}

rc=0
sync_out || rc=1
echo final_rc=$rc
exit "$rc"
