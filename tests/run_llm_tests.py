#!/usr/bin/env python3
"""
Test runner for LLM-related tests.
Provides a simple interface to run different test categories.

Run from project root or tests/ directory.
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path

# Ensure we're in the project root for relative paths to work
if Path.cwd().name == "tests":
    os.chdir("..")
    print("Changed to project root directory")

def run_unit_tests():
    """Run unit tests for LLM functionality."""
    print("Running LLM unit tests...")
    cmd = ["python", "-m", "pytest", "tests/unit/test_llm_config.py", "-v"]
    return subprocess.run(cmd).returncode == 0

def run_integration_tests():
    """Run integration tests for LLM functionality."""
    print("Running LLM integration tests...")
    cmd = ["python", "-m", "pytest", "tests/integration/test_llm_batching.py", "-v"]
    return subprocess.run(cmd).returncode == 0

def run_performance_test(model="glm-4.5", samples=3):
    """Run performance benchmark for specified model."""
    print(f"Running performance test for {model}...")
    cmd = ["python", "tests/test_llm_performance.py", "--model", model, "--samples", str(samples)]
    return subprocess.run(cmd).returncode == 0

def main():
    parser = argparse.ArgumentParser(description="LLM Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")  
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--model", default="glm-4.5", help="Model for performance tests")
    parser.add_argument("--samples", type=int, default=3, help="Samples for performance tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    
    args = parser.parse_args()
    
    success = True
    
    if args.all or args.unit:
        if not run_unit_tests():
            success = False
            print("❌ Unit tests failed")
        else:
            print("✅ Unit tests passed")
    
    if args.all or args.integration:
        if not run_integration_tests():
            success = False
            print("❌ Integration tests failed") 
        else:
            print("✅ Integration tests passed")
    
    if args.all or args.performance:
        if not run_performance_test(args.model, args.samples):
            success = False
            print("❌ Performance test failed")
        else:
            print("✅ Performance test completed")
    
    if not any([args.unit, args.integration, args.performance, args.all]):
        print("No test category specified. Use --help for options.")
        return 1
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())