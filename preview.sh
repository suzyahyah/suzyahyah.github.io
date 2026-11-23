#!/usr/bin/env bash
# Live markdown preview while writing posts.
#
# Usage:
#   ./preview.sh <file.md>   glow re-renders file in this pane on every save
#   ./preview.sh site        jekyll serve with livereload + drafts, opens browser (renders math/layout faithfully)
set -euo pipefail

target="${1:-}"
poll_seconds="${PREVIEW_POLL_SECONDS:-1}"

if [[ -z "$target" ]]; then
  sed -n '2,6p' "$0"
  exit 1
fi

if [[ "$target" == "site" ]]; then
  (sleep 4 && open "http://localhost:4000") &
  bundle exec jekyll serve --livereload --incremental --drafts
  exit
fi

# ponytail: mtime polling (no fswatch/entr dep); ceiling = 1s latency. Upgrade: brew install entr.
last_mtime=""
while true; do
  mtime=$(stat -f %m "$target")
  if [[ "$mtime" != "$last_mtime" ]]; then
    clear
    glow -s dark -w "$(tput cols)" "$target"
    last_mtime="$mtime"
  fi
  sleep "$poll_seconds"
done
