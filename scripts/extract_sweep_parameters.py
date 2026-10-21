#!/usr/bin/env python3
"""
Extract load sweep parameters (chi_max, capacity, 0.6x rate) from completed sweeps.
Run after all fleet-size sweeps are complete.

Usage:
    python scripts/extract_sweep_parameters.py
"""

import json
import statistics
from pathlib import Path

def extract_from_sweep(sweep_dir: Path) -> dict | None:
    """Extract parameters from a completed sweep directory."""
    jsonl_file = sweep_dir / "load_sweep.jsonl"

    if not jsonl_file.exists():
        return None

    # Read results
    results = []
    with open(jsonl_file) as f:
        for line in f:
            if line.strip():
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not results:
        return None

    # Filter for fixed storage rule (baseline)
    fixed_results = [r for r in results if r.get("storage_mode") == "fixed"]
    if not fixed_results:
        return None

    # Group by rate
    by_rate = {}
    for r in fixed_results:
        rate = r.get("arrival_rate")
        if rate not in by_rate:
            by_rate[rate] = []
        by_rate[rate].append(r)

    # Calculate means per rate
    metrics = {}
    for rate in sorted(by_rate.keys()):
        items = by_rate[rate]
        throughputs = [x.get("throughput", 0) for x in items]
        slopes = [x.get("backlog_slope", 0) for x in items]

        mean_throughput = statistics.mean(throughputs) if throughputs else 0
        mean_slope = statistics.mean(slopes) if slopes else 0

        metrics[rate] = {
            "throughput": mean_throughput,
            "backlog_slope": mean_slope,
            "num_seeds": len(items),
        }

    # Find capacity (highest throughput)
    capacity = max([m["throughput"] for m in metrics.values()]) if metrics else 0

    # Find chi_max (maximum healthy backlog slope)
    # From thesis: runs below capacity have healthy slopes
    healthy_slopes = []
    capacity_threshold = capacity * 0.95

    for rate, m in metrics.items():
        if m["throughput"] >= capacity_threshold:
            healthy_slopes.append(m["backlog_slope"])

    chi_max = max(healthy_slopes) if healthy_slopes else 1.0

    # Rate at 0.6 of capacity
    rate_at_0_6 = capacity * 0.6

    return {
        "capacity": capacity,
        "chi_max": chi_max,
        "rate_at_0_6": rate_at_0_6,
        "metrics_by_rate": metrics,
    }

def main():
    repo_root = Path(__file__).parent.parent
    dir_root = repo_root / "DIR"

    print("\n" + "=" * 80)
    print("LOAD SWEEP PARAMETER EXTRACTION")
    print("=" * 80)

    results = {}

    # Check 8-agent sweep (single file in DIR root)
    sweep_8 = dir_root / "load_sweep.jsonl"
    if sweep_8.exists():
        print(f"\n8-agent sweep (existing)...")
        params = extract_from_sweep(dir_root)
        if params:
            results[8] = params
            print(f"  ✓ Capacity: {params['capacity']:.4f} tasks/step")
            print(f"  ✓ Chi_max: {params['chi_max']:.4f}")
            print(f"  ✓ Rate at 0.6x: {params['rate_at_0_6']:.4f} tasks/step")
        else:
            print(f"  ✗ Could not extract parameters")

    # Check other fleet sizes
    for agents in [16, 32, 64]:
        sweep_dir = dir_root / f"sweep_{agents}_agents"
        if sweep_dir.exists() and (sweep_dir / "load_sweep.jsonl").exists():
            print(f"\n{agents}-agent sweep...")
            params = extract_from_sweep(sweep_dir)
            if params:
                results[agents] = params
                print(f"  ✓ Capacity: {params['capacity']:.4f} tasks/step")
                print(f"  ✓ Chi_max: {params['chi_max']:.4f}")
                print(f"  ✓ Rate at 0.6x: {params['rate_at_0_6']:.4f} tasks/step")
            else:
                print(f"  ✗ Could not extract parameters")
        else:
            print(f"\n⏳ {agents}-agent sweep not ready")

    if not results:
        print("\n⚠ No sweep results found. Run the sweeps first.")
        return

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Fleet Size Parameters")
    print("=" * 80)
    print("\n| Agents | Capacity | Chi_max | Rate at 0.6x |")
    print("|--------|----------|---------|-------------|")

    for agents in sorted(results.keys()):
        r = results[agents]
        print(f"| {agents:6} | {r['capacity']:8.4f} | {r['chi_max']:7.4f} | {r['rate_at_0_6']:11.4f} |")

    # Save to file
    output_file = dir_root / "fleet_parameters.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Parameters saved to {output_file}")

    # Print next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print(f"""
1. Review the parameters above
2. Update configs with arrival_rate from "Rate at 0.6x" for each fleet size
3. Set chi_max in code from the extracted value
4. Start the long training runs (step 5 of thesis plan)
""")

if __name__ == "__main__":
    main()
