#!/usr/bin/env bash
# Download videos into the local dataset tree used by this repo (see setup_local_training_data.py).
# Docs: https://github.com/ytdl-org/youtube-dl/
#
# Install: pip install youtube-dl   (or: pip install -r requirements-dataset.txt)
#
# Examples:
#   ./scripts/download_youtube_dataset.sh 'https://www.youtube.com/watch?v=VIDEO_ID'
#   ./scripts/download_youtube_dataset.sh --urls-file scripts/youtube_urls.example.txt
#   HTTPS_PROXY=http://127.0.0.1:7890 ./scripts/download_youtube_dataset.sh '<url>'
#   ./scripts/download_youtube_dataset.sh --proxy http://127.0.0.1:7890 '<url>'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_DST="${REPO_ROOT}/qevd-fit-coach-data/videos/long_range"

DST="${DEFAULT_DST}"
URLS_FILE=""
EXTRA_ARGS=()
PROXY="${YOUTUBE_DL_PROXY:-${HTTPS_PROXY:-${HTTP_PROXY:-}}}"

usage() {
  sed -n '1,40p' "$0" | tail -n +2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dst|-o)
      DST="${2:?}"
      shift 2
      ;;
    --urls-file|-f)
      URLS_FILE="${2:?}"
      shift 2
      ;;
    --proxy)
      PROXY="${2:?}"
      shift 2
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! command -v youtube-dl &>/dev/null; then
  echo "youtube-dl not found. Install with: pip install youtube-dl" >&2
  exit 1
fi

mkdir -p "${DST}"

# Filename: YouTube id + title keeps files unique and traceable; rename to zero-padded stems if you match pipeline CSVs.
OUT_TMPL="${DST}/%(id)s_%(title)s.%(ext)s"

run_one() {
  local url="$1"
  # shellcheck disable=SC2086
  youtube-dl \
    -f best \
    --default-search "ytsearch" \
    --verbose \
    -o "${OUT_TMPL}" \
    ${PROXY:+--proxy "${PROXY}"} \
    "${url}"
}

if [[ -n "${URLS_FILE}" ]]; then
  if [[ ! -f "${URLS_FILE}" ]]; then
    echo "File not found: ${URLS_FILE}" >&2
    exit 1
  fi
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    run_one "${line}"
  done < "${URLS_FILE}"
elif [[ ${#EXTRA_ARGS[@]} -eq 0 ]]; then
  echo "Pass at least one URL/search, or use --urls-file PATH" >&2
  usage
  exit 1
else
  for url in "${EXTRA_ARGS[@]}"; do
    run_one "${url}"
  done
fi

echo "Done. Videos under: ${DST}"
