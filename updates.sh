#!/bin/bash
# loc.txt format
# [src_path1] [dest_path1]
# [src_path2] [dest_path2]
# ...

LOC_PATH="loc.txt"
if [ ! -f "$LOC_PATH" ]; then
    echo "Error: File not found: $LOC_PATH"
    exit 1
fi

while IFS= read -r line; do
    if [[ -z "$line" || "$line" =~ ^# ]]; then
        continue
    fi
    read -r -a paths <<< "$line"
    SRC_PATH="${paths[0]}"
    DEST_PATH="${paths[1]}" # path relative to repo root

    if [ -z "$SRC_PATH" ] || [ -z "$DEST_PATH" ]; then
        echo "Invalid line: $line"
        continue
    fi
    
    if [ ! -e "$SRC_PATH" ]; then
        echo "Error: File not found: $SRC_PATH"
        continue
    fi

    DEST_DIR="$(dirname "$DEST_PATH")"
    if [ ! -d "$DEST_DIR" ]; then
        echo "Creating directory: $DEST_DIR"
        mkdir -p "$DEST_DIR"
    fi

    cp "$SRC_PATH" "$DEST_PATH"
    if [ $? -ne 0 ]; then
        echo "Error: Copy failed: $SRC_PATH"
    fi
done < "$LOC_PATH"
