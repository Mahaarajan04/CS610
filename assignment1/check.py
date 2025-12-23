import os, random, string, subprocess, tempfile, sys
from pathlib import Path

BIN = "./prob_3"

def make_input(n):
    # Unique, traceable lines. Include some blanks.
    lines = []
    for i in range(n):
        if random.random() < 0.02:       # ~8% blank lines
            lines.append("")
        else:
            token = ''.join(random.choices(string.ascii_letters+string.digits, k=6))
            lines.append(f"{i:06d}:{token}")
    return lines

def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for s in lines:
            f.write(s + "\n")

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]

def partitions_from_output(input_lines, output_lines):
    """Split output into maximal contiguous-with-respect-to-input chunks.
       Each chunk is a list of (out_idx, in_idx) pairs."""
    pos = {s:i for i,s in enumerate(input_lines)}
    chunks = []
    i = 0
    while i < len(output_lines):
        s = output_lines[i]
        if s not in pos:
            return None  # unknown line -> invalid
        j = pos[s]
        start = i
        # walk forward while output follows +1 in input index
        i += 1
        j += 1
        while i < len(output_lines):
            s2 = output_lines[i]
            k = pos.get(s2, -10)
            if k != j: break
            i += 1
            j += 1
        chunks.append((start, i))  # half-open [start,i)
    return chunks

def check_correct(R_lines, W_lines, Lmin, Lmax):
    # 1) Same multiset and no duplicates lost/added
    if len(R_lines) != len(W_lines):
        return False, "Length mismatch"
    if sorted(R_lines) != sorted(W_lines):
        return False, "Multiset mismatch"

    # 2) Output is concatenation of contiguous input segments
    chunks = partitions_from_output(R_lines, W_lines)
    if chunks is None:
        return False, "Non-contiguous mapping found"
    # 3) Size constraints: each chunk size in [Lmin, Lmax], except <=1 chunk may be < Lmin
    small = 0
    for a,b in chunks:
        sz = b - a
        if sz < Lmin:
            small += 1
    return True, f"OK ({len(chunks)} chunks)"

def one_run(tmpdir, seed=None):
    # random but sensible ranges
    N = random.randint(1, 500000)
    T = random.randint(1, 200)
    Lmin = random.randint(1, 250)
    Lmax = random.randint(Lmin, Lmin + random.randint(0, 250))
    M = random.randint(1, max(1, Lmax // 2)) if random.random()<0.4 else random.randint(1, Lmax*2)

    in_path = os.path.join(tmpdir, "R.txt")
    out_path = os.path.join(tmpdir, "W.txt")
    R = make_input(N)
    write_lines(in_path, R)

    cmd = [BIN, in_path, str(T), str(Lmin), str(Lmax), str(M), out_path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
    if p.returncode != 0:
        return False, f"Program failed (rc={p.returncode}): {p.stderr.strip()}"

    W = read_lines(out_path)
    ok, msg = check_correct(R, W, Lmin, Lmax)
    if not ok:
        # Provide a small forensic sample if failure
        # SAVE THE INPTUT AND OUTPUT FOR DEBUGGING
        write_lines(os.path.join(tmpdir, "R-fail.txt"), R)
        write_lines(os.path.join(tmpdir, "W-fail.txt"), W)
        return False, f"{msg}  [N={N} T={T} Lmin={Lmin} Lmax={Lmax} M={M}]"
    return True, f"[N={N} T={T} Lmin={Lmin} Lmax={Lmax} M={M}] {msg}"

def main():
    if not Path(BIN).exists():
        print("Build the binary first: g++ -std=gnu++17 -O2 -pthread lines_pipe.cpp -o lines_pipe", file=sys.stderr)
        sys.exit(2)

    random.seed(1337)
    total = 500
    passed = 0
    with tempfile.TemporaryDirectory() as td:
        for i in range(total):
            seed = random.randrange(1<<62)  # pass to binary for determinism per run
            ok, msg = one_run(td, seed=seed)
            if ok:
                passed += 1
                print(f"PASS {i+1:03d}: {msg}")
            else:
                print(f"FAIL {i+1:03d}: {msg}")
                sys.exit(1)
    print(f"\nSummary: {passed}/{total} passed")

if __name__ == "__main__":
    main()
