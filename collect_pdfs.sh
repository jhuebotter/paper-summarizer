#!/usr/bin/env bash
# Copy every PDF under <source_dir> into one flat <dest_dir>.
#
# Usage: ./collect_pdfs.sh <source_dir> [dest_dir]
#
# Files are named "<parent_folder>__<original_filename>". If that name is
# already taken by a *different* file, a numeric suffix is added (__2, __3, ...);
# identical files are not copied twice. Existing files are never overwritten.
# Handles spaces, leading/trailing whitespace, and other odd characters in
# names, and matches .pdf case-insensitively.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <source_dir> [dest_dir]" >&2
    exit 2
fi

SRC="$1"
DEST="${2:-./collected_pdfs}"

if [[ ! -d "$SRC" ]]; then
    echo "Source directory not found: $SRC" >&2
    exit 1
fi

mkdir -p "$DEST"
DEST_ABS="$(cd "$DEST" && pwd)"

copied=0
skipped=0
while IFS= read -r -d '' pdf; do
    # Never re-collect files that already live in the destination.
    case "$(cd "$(dirname "$pdf")" && pwd)" in "$DEST_ABS"*) continue ;; esac

    parent="$(basename "$(dirname "$pdf")")"
    filename="$(basename "$pdf")"
    base="${parent}__${filename}"
    target="$DEST/$base"

    n=2
    while [[ -e "$target" ]]; do
        if cmp -s "$pdf" "$target"; then
            target=""
            break
        fi
        target="$DEST/${base%.*}__${n}.${base##*.}"
        n=$((n + 1))
    done

    if [[ -z "$target" ]]; then
        skipped=$((skipped + 1))
        continue
    fi
    cp -p "$pdf" "$target"
    copied=$((copied + 1))
done < <(find "$SRC" -type f -iname '*.pdf' ! -name '._*' -print0)

echo "Done. Copied $copied PDFs to $DEST ($skipped identical duplicates skipped)."
