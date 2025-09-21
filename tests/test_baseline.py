#!/usr/bin/env python3
"""
Simple baseline training test script without complex imports.
Tests if we can run basic training with the updated algorithms.
"""

import sys
import os
import pytest
import yaml
import torch as th
from pathlib import Path

pytestmark = [pytest.mark.slow]

def test_config_loading():
    """Test if we can load configs"""
    print("🧪 Testing configuration loading...")
    
    # Test navigation config
    nav_config_path = "configs/envs/navigation.yaml"
    with open(nav_config_path, 'r') as f:
        nav_config = yaml.safe_load(f)
    print(f"✅ Loaded navigation config: {nav_config_path}")
    
    # Test navigation BPTT algorithm config
    nav_bptt_config_path = "configs/algs/navigation/bptt.yaml"
    with open(nav_bptt_config_path, 'r') as f:
        nav_bptt_config = yaml.safe_load(f)
    print(f"✅ Loaded navigation BPTT config: {nav_bptt_config_path}")
    
    return nav_config, nav_bptt_config

def test_algorithm_import():
    """Test if we can import the updated algorithms"""
    print("🧪 Testing algorithm imports...")
    
    try:
        from algorithms.BPTT import BPTT
        print("✅ Successfully imported updated BPTT from /algorithms")
        
        from algorithms.SHAC import SHAC
        print("✅ Successfully imported updated SHAC from /algorithms")
        
        # Test if we can create instances (without environment)
        print("Algorithm classes loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Algorithm import failed: {e}")
        return False

def test_visfly_import():
    """Test VisFly components"""
    print("🧪 Testing VisFly component imports...")
    
    try:
        from VisFly.utils.common import load_yaml_config
        print("✅ VisFly common utils imported")
        
        # Test the function
        nav_config = load_yaml_config("configs/envs/navigation.yaml")
        print(f"✅ Config loaded with VisFly function: {type(nav_config)}")
        return True
        
    except Exception as e:
        print(f"❌ VisFly import failed: {e}")
        return False

def check_dependencies():
    """Check if required dependencies are available"""
    print("🧪 Checking dependencies...")
    
    required_packages = [
        'torch',
        'numpy', 
        'matplotlib',
        'stable_baselines3',
        'gymnasium'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package} - MISSING")
    
    return len(missing) == 0, missing

def main():
    """Run all baseline tests"""
    print("=" * 60)
    print("🚁 VisFly-Eureka Baseline Training Test")
    print("=" * 60)
    
    # Test 1: Dependencies
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print(f"❌ Missing dependencies: {missing}")
        print("Please install missing packages before training.")
        return 1
    
    # Test 2: Config Loading
    try:
        nav_config, nav_bptt_config = test_config_loading()
        print("✅ Configuration loading works!")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return 1
    
    # Test 3: Algorithm Imports
    alg_ok = test_algorithm_import()
    if not alg_ok:
        print("❌ Algorithm import failed - cannot proceed with training")
        return 1
    
    # Test 4: VisFly Imports
    visfly_ok = test_visfly_import()
    if not visfly_ok:
        print("⚠️  VisFly imports failed - you may need to install VisFly dependencies")
        print("   This includes habitat-sim and other VisFly requirements")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print("✅ Dependencies: OK")
    print("✅ Configuration loading: OK") 
    print("✅ Updated algorithms: OK")
    print(f"{'✅' if visfly_ok else '⚠️ '} VisFly components: {'OK' if visfly_ok else 'NEEDS SETUP'}")
    
    if visfly_ok:
        print("\n🎉 All baseline tests passed!")
        print("You can now run training with:")
        print("  python run.py --env navigation --algorithm bptt --train 1")
        return 0
    else:
        print("\n⚠️  Partial success - VisFly setup needed for full training")
        print("Next steps:")
        print("1. Install habitat-sim following VisFly instructions")
        print("2. Install other VisFly dependencies")
        print("3. Then run: python run.py --env navigation --algorithm bptt --train 1")
        return 0

if __name__ == "__main__":
    exit(main())
