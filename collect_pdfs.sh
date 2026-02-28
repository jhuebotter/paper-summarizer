#!/usr/bin/env bash
# Usage: ./collect_pdfs.sh <source_dir> <dest_dir>

SRC="${1:-.}"
DEST="${2:-./collected_pdfs}"

mkdir -p "$DEST"

find "$SRC" -type f -name "*.pdf" | while read -r pdf; do
    # Use basename to flatten; add parent dir prefix to avoid collisions
    parent=$(basename "$(dirname "$pdf")")
    filename=$(basename "$pdf")
    cp "$pdf" "$DEST/${parent}__${filename}"
done

echo "Done. $(ls "$DEST" | wc -l) PDFs copied to $DEST"
