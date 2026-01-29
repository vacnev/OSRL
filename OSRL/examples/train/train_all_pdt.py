#!/usr/bin/env python3
"""
Script to run PDT training sweeps sequentially.
Translated from pdt_bullet_sweep.yaml and pdt_gymnasium_sweep.yaml
"""

import subprocess
import sys
from itertools import product

def run_command(task, seed, update_steps, device="cuda", project="OSRL"):
    """Run a single training command."""
    cmd = [
        "python3", "-m", "examples.train.train_pdt",
        "--project", project,
        "--device", device,
        "--task", task,
        "--seed", str(seed),
        "--update_steps", str(update_steps),
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)

def run_bullet_sweep():
    """Run the bullet environments sweep."""
    print("\n" + "="*80)
    print("STARTING BULLET ENVIRONMENTS SWEEP")
    print("="*80)
    
    tasks = [
        "OfflineCarCircle-v0",
        "OfflineAntCircle-v0", 
        "OfflineDroneCircle-v0",
        "OfflineBallCircle-v0"
    ]
    seeds = [0, 10, 20, 30, 40]
    update_steps = 100_000
    
    total = len(tasks) * len(seeds)
    current = 0
    failed = []
    
    for task, seed in product(tasks, seeds):
        current += 1
        print(f"\n[{current}/{total}] Task: {task}, Seed: {seed}")
        
        success = run_command(task, seed, update_steps)
        if not success:
            failed.append((task, seed))
    
    return failed

def run_gymnasium_sweep():
    """Run the gymnasium environments sweep."""
    print("\n" + "="*80)
    print("STARTING GYMNASIUM ENVIRONMENTS SWEEP")
    print("="*80)
    
    tasks = [
        "OfflineSwimmerVelocityGymnasium-v1",
        "OfflineHopperVelocityGymnasium-v1",
        "OfflineHalfCheetahVelocityGymnasium-v1",
        "OfflineCarButton1Gymnasium-v0",
        "OfflineCarButton2Gymnasium-v0",
        "OfflineCarPush1Gymnasium-v0",
        "OfflineCarPush2Gymnasium-v0",
        "OfflineCarGoal1Gymnasium-v0",
        "OfflineCarGoal2Gymnasium-v0"
    ]
    seeds = [0, 10, 20, 30, 40]
    update_steps = 200_000
    
    total = len(tasks) * len(seeds)
    current = 0
    failed = []
    
    for task, seed in product(tasks, seeds):
        current += 1
        print(f"\n[{current}/{total}] Task: {task}, Seed: {seed}")
        
        success = run_command(task, seed, update_steps)
        if not success:
            failed.append((task, seed))
    
    return failed

def main():
    """Run all sweeps."""
    print("Starting PDT training sweeps...")
    
    # Run bullet sweep
    bullet_failed = run_bullet_sweep()
    
    # Run gymnasium sweep
    gymnasium_failed = run_gymnasium_sweep()
    
    # Summary
    print("\n" + "="*80)
    print("SWEEP COMPLETE")
    print("="*80)
    
    all_failed = bullet_failed + gymnasium_failed
    
    if all_failed:
        print(f"\n⚠️  {len(all_failed)} runs failed:")
        for task, seed in all_failed:
            print(f"  - {task} (seed={seed})")
        sys.exit(1)
    else:
        print("\n✅ All runs completed successfully!")

if __name__ == "__main__":
    main()
