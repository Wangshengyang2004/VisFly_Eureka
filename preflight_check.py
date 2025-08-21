#!/usr/bin/env python3
"""
Pre-flight Check for VisFly-Eureka

Run this script before main.py to verify all dependencies and configurations are ready.
"""

import sys
import os
from pathlib import Path
import importlib.util

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check(condition, success_msg, fail_msg):
    """Print check result with color"""
    if condition:
        print(f"{GREEN}✓{RESET} {success_msg}")
        return True
    else:
        print(f"{RED}✗{RESET} {fail_msg}")
        return False

def main():
    print("=" * 60)
    print("VISFLY-EUREKA PRE-FLIGHT CHECK")
    print("=" * 60)
    
    all_good = True
    
    # 1. Check Python version
    python_version = sys.version_info
    all_good &= check(
        python_version >= (3, 9),
        f"Python {python_version.major}.{python_version.minor} OK",
        f"Python {python_version.major}.{python_version.minor} too old (need >= 3.9)"
    )
    
    # 2. Check critical Python packages
    packages = {
        'torch': 'PyTorch',
        'yaml': 'PyYAML',
        'omegaconf': 'OmegaConf',
        'hydra': 'Hydra',
        'openai': 'OpenAI',
        'numpy': 'NumPy',
        'gymnasium': 'Gymnasium',
    }
    
    print("\n📦 Python Dependencies:")
    for pkg, name in packages.items():
        spec = importlib.util.find_spec(pkg)
        all_good &= check(spec is not None, f"{name} installed", f"{name} NOT FOUND")
    
    # 3. Check CUDA availability (warning only)
    print("\n🖥️ GPU Configuration:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"{GREEN}✓{RESET} CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"{YELLOW}⚠{RESET} CUDA not available - will use CPU (slower)")
    except:
        print(f"{YELLOW}⚠{RESET} Could not check CUDA availability")
    
    # 4. Check VisFly installation
    print("\n🚁 VisFly Installation:")
    visfly_path = Path(__file__).parent / "VisFly"
    all_good &= check(
        visfly_path.exists(),
        f"VisFly directory found at {visfly_path}",
        f"VisFly directory NOT FOUND at {visfly_path}"
    )
    
    # Check if VisFly can be imported
    if visfly_path.exists():
        sys.path.insert(0, str(visfly_path))
        try:
            from envs.NavigationEnv import NavigationEnv
            check(True, "VisFly NavigationEnv importable", "")
        except ImportError as e:
            all_good &= check(False, "", f"Cannot import VisFly: {e}")
    
    # 5. Check habitat-sim
    print("\n🏠 Habitat-Sim:")
    try:
        import habitat_sim
        check(True, "habitat-sim installed", "")
    except ImportError:
        all_good &= check(False, "", "habitat-sim NOT INSTALLED - required for VisFly")
    
    # 6. Check dataset
    print("\n📊 Datasets:")
    dataset_path = visfly_path / "datasets" / "visfly-beta"
    if dataset_path.exists():
        check(True, f"visfly-beta dataset found", "")
    else:
        print(f"{YELLOW}⚠{RESET} visfly-beta dataset not found at {dataset_path}")
        print(f"  Download from: https://huggingface.co/datasets/LiFanxing/visfly-beta")
    
    # 7. Check API configuration
    print("\n🔑 API Configuration:")
    api_config_path = Path(__file__).parent / "configs" / "api_keys.yaml"
    
    if api_config_path.exists():
        try:
            import yaml
            with open(api_config_path, 'r') as f:
                api_config = yaml.safe_load(f)
            
            # Check for API keys (don't print them!)
            has_openai = api_config.get('openai', {}).get('api_key', '').startswith('sk-') or \
                         api_config.get('openai', {}).get('api_key', '') == 'YOUR_OPENAI_API_KEY_HERE'
            has_yunwu = api_config.get('yunwu', {}).get('api_key', '').startswith('sk-') or \
                        api_config.get('yunwu', {}).get('api_key', '') == 'YOUR_YUNWU_API_KEY_HERE'
            
            if has_openai and 'YOUR_OPENAI_API_KEY_HERE' not in api_config.get('openai', {}).get('api_key', ''):
                check(True, "OpenAI API key configured", "")
            else:
                print(f"{YELLOW}⚠{RESET} OpenAI API key not configured in api_keys.yaml")
            
            if has_yunwu and 'YOUR_YUNWU_API_KEY_HERE' not in api_config.get('yunwu', {}).get('api_key', ''):
                check(True, "Yunwu API key configured", "")
            else:
                print(f"{YELLOW}⚠{RESET} Yunwu API key not configured in api_keys.yaml")
                
        except Exception as e:
            print(f"{RED}✗{RESET} Error reading api_keys.yaml: {e}")
    else:
        print(f"{RED}✗{RESET} api_keys.yaml not found")
        print(f"  Create it by copying: cp configs/api_keys.example.yaml configs/api_keys.yaml")
        all_good = False
    
    # Check environment variables as fallback
    print("\n🌍 Environment Variables:")
    env_vars = ['OPENAI_API_KEY', 'YUNWU_API_KEY', 'ANTHROPIC_API_KEY']
    for var in env_vars:
        if os.getenv(var):
            print(f"{GREEN}✓{RESET} {var} is set")
        else:
            print(f"{YELLOW}⚠{RESET} {var} not set (optional if using config file)")
    
    # 8. Check Hydra configuration
    print("\n⚙️ Hydra Configuration:")
    config_path = Path(__file__).parent / "configs"
    config_file = config_path / "config.yaml"
    all_good &= check(
        config_file.exists(),
        f"Main config.yaml found",
        f"config.yaml NOT FOUND at {config_file}"
    )
    
    # 9. Check output directories
    print("\n📁 Output Directories:")
    outputs_dir = Path(__file__).parent / "outputs"
    if not outputs_dir.exists():
        outputs_dir.mkdir(parents=True, exist_ok=True)
        print(f"{GREEN}✓{RESET} Created outputs directory")
    else:
        print(f"{GREEN}✓{RESET} Outputs directory exists")
    
    # Final summary
    print("\n" + "=" * 60)
    if all_good:
        print(f"{GREEN}✅ ALL CRITICAL CHECKS PASSED!{RESET}")
        print("\nYou can now run:")
        print("  python main.py")
        print("\nOr with custom settings:")
        print("  python main.py optimization.iterations=3 optimization.samples=8")
    else:
        print(f"{RED}❌ SOME CRITICAL CHECKS FAILED{RESET}")
        print("\nPlease fix the issues above before running main.py")
        print("\nKey setup steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install VisFly and habitat-sim (see README)")
        print("3. Download dataset: https://huggingface.co/datasets/LiFanxing/visfly-beta")
        print("4. Configure API keys: cp configs/api_keys.example.yaml configs/api_keys.yaml")
        sys.exit(1)

if __name__ == "__main__":
    main()