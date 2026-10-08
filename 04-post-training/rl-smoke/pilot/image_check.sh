#!/bin/bash
# Pilot leak check, part 1: look inside each pool task's image (local Docker, network off) for answers the agent could
# dig up instead of solving: Git history and dangling objects, test or solution files, build caches, and file times.
# One report per task in image_check/<task>.txt; images are removed after each check.
#
#   ./image_check.sh [pool.txt]
cd "$(dirname "$0")"
T=~/.cache/harbor/terminal-bench_terminal-bench-2/terminal-bench-2
mkdir -p image_check
PROBE='
echo "## whoami: $(whoami)  workdir: $(pwd)"
echo "## git repos (outside /usr, /proc, /sys)"
for g in $(find / -xdev -name .git -not -path "/proc/*" -not -path "/usr/*" -not -path "/sys/*" 2>/dev/null); do
  d=$(dirname $g); echo "repo $d"
  if command -v git >/dev/null; then
    echo "  commits (all refs): $(git -C $d log --all --oneline 2>/dev/null | wc -l)  reflog: $(git -C $d reflog 2>/dev/null | wc -l)  stashes: $(git -C $d stash list 2>/dev/null | wc -l)"
    echo "  dangling: $(git -C $d fsck --lost-found 2>/dev/null | wc -l)  branches: $(git -C $d branch -a 2>/dev/null | tr -d " " | tr "\n" ",")"
    git -C $d log --all --format="  %h %ad %s" --date=short 2>/dev/null | head -8
  else echo "  (no git binary)"; fi
done
echo "## test / solution / answer-like files"
find / -xdev \( -iname "*solution*" -o -iname "solve.sh" -o -iname "test_outputs.py" -o -iname "*expected*" -o -iname "*answer*" -o -iname "oracle*" -o -path "/tests/*" \) \
  -not -path "/proc/*" -not -path "/usr/*" -not -path "/sys/*" -not -path "*/site-packages/*" -not -path "*/dist-packages/*" -not -path "*/node_modules/*" 2>/dev/null | head -40
echo "## caches"
for c in /root/.cache /root/.cargo /root/.npm /tmp /var/tmp /app/build /app/.pytest_cache; do [ -e $c ] && echo "$c $(du -sh $c 2>/dev/null | cut -f1) $(find $c -type f 2>/dev/null | wc -l) files"; done
find /app -xdev \( -name "__pycache__" -o -name "*.o" -o -name "*.pyc" -o -name "*.gcda" -o -name "*.log" \) 2>/dev/null | head -15
echo "## /app listing (full times)"
ls -la --time-style=full-iso /app 2>/dev/null | head -40
find /app -xdev -type f 2>/dev/null | wc -l | sed "s/^/app files: /"
'
for t in $(cat ${1:-pool.txt}); do
  img=$(grep -h docker_image $T/$t/task.toml | cut -d'"' -f2)
  echo "$(date +%T) $t ($img)"
  docker pull -q $img > /dev/null 2>&1 || { echo "pull failed" > image_check/$t.txt; continue; }
  size=$(docker image inspect $img --format '{{.Size}}' | awk '{printf "%.1f GB", $1/1e9}')
  { echo "# $t  $img  $size"; timeout 600 docker run --rm --network none --entrypoint sh $img -c "$PROBE" 2>&1; } > image_check/$t.txt
  docker image rm -f $img > /dev/null 2>&1
done
echo "$(date +%T) done"
