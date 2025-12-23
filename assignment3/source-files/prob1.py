import os
import subprocess
import statistics

CPP_FILE = "stencil_driver.cpp"
EXE_FILE = "stencil_exec"

SZ_LIST = [64, 128, 256, 512, 1024]
UNROLL_LIST = [2, 4, 6, 8, 16, 32]
N_RUNS = 5

# Read base code once
with open(CPP_FILE, 'r') as f:
    base_code = f.read()

results = []

def run_variant(sz, uf=None, parallel=False, baseline=False):
    """
    Returns list of N_RUNS execution times (in ms) for the specified variant:
    - baseline: run baseline kernel only
    - if not baseline: runs unrolled (parallel or non-parallel)
    """
    code = base_code
    code = code.replace("#define SZ 64", f"#define SZ {sz}")

    if baseline:
        # Leave PAR and UNROLL_FACTOR unchanged, doesn't matter
        pass
    else:
        code = code.replace("#define UNROLL_FACTOR 4", f"#define UNROLL_FACTOR {uf}")
        if parallel:
            code = code.replace("#define PAR 1", f"#define PAR 1")
        else:
            code = "\n".join(
                line for line in code.splitlines()
                if not line.strip().startswith("#define PAR")
            )

    # Write modified cpp
    with open(CPP_FILE, 'w') as f:
        f.write(code)

    # Compile
    compile_cmd = f"g++ -O3 -fopenmp -march=native {CPP_FILE} -o {EXE_FILE}"
    if subprocess.call(compile_cmd, shell=True) != 0:
        return None

    # Run N times
    times = []
    for _ in range(N_RUNS):
        try:
            out = subprocess.check_output(f"./{EXE_FILE}", shell=True, text=True)
            for line in out.splitlines():
                if baseline and "Baseline kernel time" in line:
                    times.append(int(line.split(":")[1].split()[0]))
                    break
                elif not baseline and "Unrolled kernel time" in line:
                    times.append(int(line.split(":")[1].split()[0]))
                    break
        except subprocess.CalledProcessError:
            return None

    return times if len(times) == N_RUNS else None


# Main loop
for sz in SZ_LIST:
    # Run BASELINE first for each SZ
    base_times = run_variant(sz, baseline=True)
    base_avg = round(statistics.mean(base_times), 2) if base_times else None

    for uf in UNROLL_LIST:
        # Unrolled Non‑Parallel
        nonpar_times = run_variant(sz, uf=uf, parallel=False)
        nonpar_avg = round(statistics.mean(nonpar_times), 2) if nonpar_times else None

        # Unrolled Parallel
        par_times = run_variant(sz, uf=uf, parallel=True)
        par_avg = round(statistics.mean(par_times), 2) if par_times else None

        results.append((sz, uf, base_avg, nonpar_avg, par_avg))
        print(f"SZ={sz:4} UNROLL={uf:2} | Base={base_avg} ms | NonPar={nonpar_avg} ms | Par={par_avg} ms")

# Final summary
print("\n======== SUMMARY TABLE ========")
print("SZ     UF   Baseline   Unrolled     Parallel")
print("---------------------------------------------")
for sz, uf, base, nonpar, par in results:
    print(f"{sz:<6} {uf:<4} {base:>8} ms  {nonpar:>8} ms  {par:>8} ms")
