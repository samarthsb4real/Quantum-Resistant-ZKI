#!/usr/bin/env python3
"""
Quantum-Resistant Hash Framework - Quick Start
Simple launcher for main functionality
"""

import sys
import subprocess

def main():
    print("Quantum-Resistant Hash Framework v2")
    print("=" * 40)
    print("1. Main Analysis")
    print("2. Interactive CLI")
    print("3. Visual Analysis")
    print("4. Complete Report")
    print("5. Transmission Demo")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    scripts = {
        '1': 'quantum_hash_prototype.py',
        '2': 'cli.py',
        '3': 'visual_analysis.py', 
        '4': 'master_report.py',
        '5': 'transmission_demo.py'
    }
    
    if choice in scripts:
        if choice == '2':
            subprocess.run([sys.executable, scripts[choice], 'interactive'])
        else:
            subprocess.run([sys.executable, scripts[choice]])
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()