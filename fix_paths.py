#!/usr/bin/env python3
"""
Fix hardcoded absolute paths in test files
"""

import os
from pathlib import Path

# Files to fix
test_files = [
    'tests/navigation_example.py',
    'tests/simple_usage_real.py', 
    'tests/real_navigation_example.py',
]

# Replacement pattern
old_pattern = """sys.path.insert(0, '/home/simonwsy/VisFly_Eureka')
sys.path.insert(0, '/home/simonwsy/VisFly_Eureka/VisFly')"""

new_pattern = """PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'VisFly'))"""

# Also need to add Path import
import_addition = "from pathlib import Path\n"

for file_path in test_files:
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        continue
        
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already fixed
    if 'PROJECT_ROOT' in content:
        print(f"✓ Already fixed: {file_path}")
        continue
    
    # Add Path import if not present
    if 'from pathlib import Path' not in content:
        # Find where to add the import (after other imports)
        lines = content.split('\n')
        import_line = -1
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_line = i
        
        if import_line >= 0:
            lines.insert(import_line + 1, 'from pathlib import Path')
            content = '\n'.join(lines)
    
    # Replace hardcoded paths
    content = content.replace(old_pattern, new_pattern)
    
    # Write back
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Fixed: {file_path}")

print("\n✅ All paths fixed!")