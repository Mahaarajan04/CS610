import sys

def replicate_file(input_file, output_file, n):
    try:
        with open(input_file, 'r') as fin:
            lines = fin.readlines()

        with open(output_file, 'w') as fout:
            for _ in range(n):
                fout.writelines(lines)

        print(f"Successfully wrote {n} copies of {input_file} to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python replicate_file.py <input_file> <output_file> <N>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    try:
        N = int(sys.argv[3])
    except ValueError:
        print("Error: N must be an integer.")
        sys.exit(1)

    replicate_file(input_file, output_file, N)

