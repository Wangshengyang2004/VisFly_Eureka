"""
Test script to verify trajectory data is saved and can be loaded correctly.
This is a simpler test that doesn't require LLM API.
"""

import numpy as np
from pathlib import Path
import tempfile
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

# Define format_trajectory_data locally to avoid import dependencies


def test_trajectory_npz_save_load():
    """Test saving and loading trajectory data as npz"""
    print("=" * 60)
    print("Test: Trajectory NPZ Save/Load")
    print("=" * 60)
    
    # Create mock trajectory data
    timesteps = 100
    num_agents = 1
    
    positions = np.random.randn(timesteps, num_agents, 3) * 2.0
    velocities = np.random.randn(timesteps, num_agents, 3) * 0.5
    orientations = np.random.randn(timesteps, num_agents, 4)
    # Normalize quaternions
    for i in range(timesteps):
        ori = orientations[i, 0]
        ori = ori / np.linalg.norm(ori)
        orientations[i, 0] = ori
    angular_velocities = np.random.randn(timesteps, num_agents, 3) * 0.1
    target = np.array([15.0, 0.0, 4.0])
    
    # Save to temporary npz file
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "test_trajectory.npz"
        
        trajectory_data = {
            "positions": positions,
            "velocities": velocities,
            "orientations": orientations,
            "angular_velocities": angular_velocities,
            "target": target,
        }
        
        np.savez_compressed(npz_path, **trajectory_data)
        print(f"Saved trajectory to {npz_path}")
        print(f"  File size: {npz_path.stat().st_size / 1024:.2f} KB")
        
        # Load and verify
        loaded = np.load(npz_path)
        print("\nLoaded trajectory data:")
        print(f"  Positions shape: {loaded['positions'].shape}")
        print(f"  Velocities shape: {loaded['velocities'].shape}")
        print(f"  Orientations shape: {loaded['orientations'].shape}")
        print(f"  Angular velocities shape: {loaded['angular_velocities'].shape}")
        print(f"  Target: {loaded['target']}")
        
        # Verify data integrity
        assert np.allclose(loaded['positions'], positions), "Positions don't match"
        assert np.allclose(loaded['velocities'], velocities), "Velocities don't match"
        assert np.allclose(loaded['orientations'], orientations), "Orientations don't match"
        assert np.allclose(loaded['angular_velocities'], angular_velocities), "Angular velocities don't match"
        assert np.allclose(loaded['target'], target), "Target doesn't match"
        
        print("\n✓ Data integrity verified!")
        
        # Test formatting for LLM
        print("\n" + "=" * 60)
        print("Test: Format trajectory data for LLM")
        print("=" * 60)
        
        # Use the format_trajectory_data function (standalone version)
        formatted = format_trajectory_data(str(npz_path))
        print("\nFormatted trajectory data (first 500 chars):")
        print(formatted[:500] + "..." if len(formatted) > 500 else formatted)
        
        # Verify formatting includes key information
        assert "Position" in formatted, "Should include position data"
        assert "Orientation" in formatted, "Should include orientation data"
        assert "Velocity" in formatted, "Should include velocity data"
        assert "Angular velocity" in formatted, "Should include angular velocity data"
        assert "Target position" in formatted, "Should include target data"
        
        print("\n✓ Formatting verification passed!")
        
        return True


def format_trajectory_data(trajectory_path: str) -> str:
    """Standalone version of format_trajectory_data for testing"""
    data = np.load(trajectory_path)
    
    # Extract arrays
    positions = data["positions"]
    velocities = data["velocities"]
    orientations = data["orientations"]
    angular_velocities = data["angular_velocities"]
    target = data.get("target")
    
    # Take mean across agents if multiple agents
    if positions.ndim == 3 and positions.shape[1] > 1:
        positions = positions.mean(axis=1)
        velocities = velocities.mean(axis=1)
        orientations = orientations.mean(axis=1)
        angular_velocities = angular_velocities.mean(axis=1)
    elif positions.ndim == 3:
        positions = positions.squeeze(1)
        velocities = velocities.squeeze(1)
        orientations = orientations.squeeze(1)
        angular_velocities = angular_velocities.squeeze(1)
    
    # Format as readable string
    timesteps = positions.shape[0]
    lines = [f"Trajectory data ({timesteps} timesteps):"]
    
    # Sample every 10 steps to keep it manageable
    sample_indices = np.linspace(0, timesteps - 1, min(20, timesteps), dtype=int)
    
    # Position trajectory
    lines.append(f"\nPosition (x, y, z) over {timesteps} steps:")
    for i in sample_indices:
        pos = positions[i]
        lines.append(f"  Step {i}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    # Orientation
    lines.append(f"\nOrientation (quaternion [w, x, y, z]) over {timesteps} steps:")
    for i in sample_indices:
        ori = orientations[i]
        lines.append(f"  Step {i}: [{ori[0]:.3f}, {ori[1]:.3f}, {ori[2]:.3f}, {ori[3]:.3f}]")
    
    # Velocity
    lines.append(f"\nVelocity (vx, vy, vz) over {timesteps} steps:")
    for i in sample_indices:
        vel = velocities[i]
        lines.append(f"  Step {i}: [{vel[0]:.3f}, {vel[1]:.3f}, {vel[2]:.3f}]")
    
    # Angular velocity
    lines.append(f"\nAngular velocity (ωx, ωy, ωz) over {timesteps} steps:")
    for i in sample_indices:
        angvel = angular_velocities[i]
        lines.append(f"  Step {i}: [{angvel[0]:.3f}, {angvel[1]:.3f}, {angvel[2]:.3f}]")
    
    if target is not None:
        if target.ndim == 1:
            target_str = f"[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]"
        else:
            target_str = f"[{target[0,0]:.3f}, {target[0,1]:.3f}, {target[0,2]:.3f}]"
        lines.append(f"\nTarget position: {target_str}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        test_trajectory_npz_save_load()
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

