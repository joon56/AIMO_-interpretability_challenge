#!/usr/bin/env bash
# Local helper for GPU pods. Pod address file: $POD_FILE (default cloud/POD; second pod: POD_FILE=cloud/POD2).
#   bash cloud/pod.sh push [paths...]   code + data -> /workspace/claude (default set below)
#   bash cloud/pod.sh run "<cmd>"       run a command in /workspace/claude
#   bash cloud/pod.sh pull [paths...]   default runs/{gen,bench,behavior,artifacts,report} -> runs/pod/
#                                       (hidden states are large: pull explicitly with runs/hidden)
# Upstream packets > ~1.4 KB to the pod are dropped (MTU black hole, 2026-10-26), so uploads go as
# base64 chunks of <=1000 bytes, one ssh session each, then are reassembled and checksum-verified.
set -euo pipefail
cd "$(dirname "$0")/.."
read -r -a POD < "${POD_FILE:-cloud/POD}"
SSH=(ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ConnectTimeout=20
     -o KexAlgorithms=curve25519-sha256 "${POD[@]}")

case "$1" in
  push)
    shift
    paths=("$@")
    [ ${#paths[@]} -gt 0 ] || paths=(src cloud experiments/train_probe.py experiments/build_pool.py
                                     data/pool/problems.jsonl data/raw/train-main-v2)
    tmp=$(mktemp -d)
    tar czf "$tmp/p.tgz" --exclude='__pycache__' "${paths[@]}"
    sum=$(sha256sum "$tmp/p.tgz" | cut -c1-64)
    base64 -w0 "$tmp/p.tgz" | split -b 1000 -d -a 4 - "$tmp/c."
    "${SSH[@]}" "rm -rf /tmp/push && mkdir -p /tmp/push"
    ls "$tmp"/c.* | xargs -P 8 -I{} sh -c 'f={}; n=$(basename $f); for t in 1 2 3; do "$@" "cat > /tmp/push/$n" < $f && break; done' _ "${SSH[@]}"
    "${SSH[@]}" "cat /tmp/push/c.* | base64 -d > /tmp/push.tgz && echo \$(sha256sum /tmp/push.tgz | cut -c1-64) && mkdir -p /workspace/claude && tar xzf /tmp/push.tgz --no-same-owner -C /workspace/claude" \
      | { read -r remote; [ "$remote" = "$sum" ] && echo "pushed $(ls "$tmp"/c.* | wc -l) chunks, sha ok" || { echo "CHECKSUM MISMATCH"; exit 1; }; }
    rm -rf "$tmp" ;;
  run)
    "${SSH[@]}" "cd /workspace/claude && { $2; }" ;;
  pull)
    shift
    what="${*:-runs/gen runs/bench runs/behavior runs/artifacts runs/report}"
    dest="${PULL_DIR:-runs/pod}"; mkdir -p "$dest"
    "${SSH[@]}" "cd /workspace/claude && tar czf - --exclude='*.texts.jsonl.gz' $what 2>/dev/null; true" \
      | tar xzf - -C "$dest" && du -sh "$dest"/runs/* 2>/dev/null ;;
esac
