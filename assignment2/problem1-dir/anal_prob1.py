import csv
from collections import defaultdict

input_file = 'results.csv'
output_file = 'averaged_results.csv'

# Mapping: (X, Y, Z) -> list of (Time, Misses)
data = defaultdict(list)

with open(input_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for row in reader:
        if not row or len(row) < 6:
            continue
        x, y, z = map(int, row[:3])
        time = int(row[4])
        misses = int(row[5])
        data[(x, y, z)].append((time, misses))

# Compute averages
averages = []
for (x, y, z), values in data.items():
    times = [v[0] for v in values]
    misses = [v[1] for v in values]
    avg_time = sum(times) / len(times)
    avg_misses = sum(misses) / len(misses)
    averages.append([x, y, z, avg_time, avg_misses])

# Write to output CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['X', 'Y', 'Z', 'Avg_Time(us)', 'Avg_L2_Misses'])
    writer.writerows(averages)

# Print best configurations
best_runtime = min(averages, key=lambda row: row[3])
best_misses = min(averages, key=lambda row: row[4])

print("\nBest configuration (lowest average runtime):")
print(f"X={best_runtime[0]}, Y={best_runtime[1]}, Z={best_runtime[2]}, Avg_Time={best_runtime[3]:.2f}us, L2_Misses={int(best_runtime[4])}")

print("\nBest configuration (lowest L2 cache misses):")
print(f"X={best_misses[0]}, Y={best_misses[1]}, Z={best_misses[2]}, Avg_Time={best_misses[3]:.2f}us, L2_Misses={int(best_misses[4])}")

