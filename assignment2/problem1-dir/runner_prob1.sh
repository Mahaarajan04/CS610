#!/bin/bash

# Path to your compiled binary
EXECUTABLE="./prob1"
OUTFILE="results.csv"

# Clear previous results
echo "X,Y,Z,Run,Time(us),L2_Misses" > "$OUTFILE"

# Function to flush the cache (writes to 100MB dummy array)
flush_cache() {
    dummy=$(mktemp)
    head -c 100000000 /dev/urandom > "$dummy"
    rm -f "$dummy"
}

# Only use power-of-two block sizes
block_sizes=(1 2 4 8 16 32)

for X in "${block_sizes[@]}"; do
  for Y in "${block_sizes[@]}"; do
    for Z in "${block_sizes[@]}"; do
      echo "Running Block Size: X=$X Y=$Y Z=$Z"

      for run in {1..5}; do
        flush_cache

        # Capture output from the executable
        output=$($EXECUTABLE $X $Y $Z)

        # Extract time and L2 misses
        time_us=$(echo "$output" | grep "Runtime" | cut -d':' -f2 | xargs)
        l2_miss=$(echo "$output" | grep "L2 Cache Misses" | cut -d':' -f2 | xargs)

        # Append to CSV
        echo "$X,$Y,$Z,$run,$time_us,$l2_miss" >> "$OUTFILE"
      done

      # Add blank line after each triple
      echo "" >> "$OUTFILE"
    done
  done
done

