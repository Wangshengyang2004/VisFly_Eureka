"""
Test video recording functionality in evaluation_worker.py specifically.

Validates that the AVC1 codec fix resolves video recording issues.
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

import pytest


class TestEvaluationWorkerVideo(unittest.TestCase):
    """Test evaluation worker video recording with AVC1 fix."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="eval_worker_video_test_"))

    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.skipif(cv2 is None, reason="OpenCV not available")
    def test_avc1_codec_fix(self):
        """Test that AVC1 codec works for video recording."""
        # Test the exact codec used in evaluation_worker.py
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video_path = self.temp_dir / "test_avc1_fix.mp4"

        # Test video creation
        video_writer = cv2.VideoWriter(
            str(video_path), fourcc, 30.0, (640, 480)
        )

        self.assertTrue(video_writer.isOpened(),
                       "AVC1 video writer should open successfully")

        # Write test frames
        for i in range(10):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            video_writer.write(frame)

        video_writer.release()

        # Verify file was created and has content
        self.assertTrue(video_path.exists(), "Video file should be created")
        self.assertGreater(video_path.stat().st_size, 1000,
                          "Video file should have substantial content")

        print(f"✓ AVC1 codec test passed: {video_path.stat().st_size} bytes")

    def test_frame_extraction_compatibility(self):
        """Test _extract_rgb_frame function with various input formats."""
        # Import the function from evaluation_worker
        import sys
        import importlib.util

        eval_worker_path = Path(__file__).parent.parent.parent / "quadro_llm" / "core" / "evaluation_worker.py"
        spec = importlib.util.spec_from_file_location("evaluation_worker", eval_worker_path)
        eval_worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_worker)

        extract_rgb_frame = eval_worker._extract_rgb_frame

        # Test standard RGB frame (most common from VisFly)
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = extract_rgb_frame(test_frame)

        self.assertIsNotNone(result, "Should extract valid RGB frame")
        self.assertEqual(result.shape, (480, 640, 3), "Should maintain correct shape")
        self.assertEqual(result.dtype, np.uint8, "Should be uint8 format")

        # Test RGBA frame (drop alpha to RGB)
        test_frame_rgba = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
        result_rgba = extract_rgb_frame(test_frame_rgba)
        self.assertIsNotNone(result_rgba, "Should handle RGBA by dropping alpha")
        self.assertEqual(result_rgba.shape, (480, 640, 3), "RGBA should be converted to RGB")

        print("✓ Frame extraction test passed")

    def test_video_recording_pipeline_simulation(self):
        """Simulate the exact video recording pipeline from evaluation_worker."""
        if cv2 is None:
            self.skipTest("OpenCV not available")

        # Simulate evaluation parameters matching evaluation_worker.py
        max_steps = 30
        video_fps = 30.0
        video_dir = self.temp_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        # Test parameters matching evaluation configuration
        episode_idx = 0
        video_enabled = True
        video_writer = None
        frames_captured = 0
        video_path = None

        # Simulate episode steps (matching evaluation_worker.py logic)
        for step_count in range(max_steps):
            # Simulate env.render() returning an RGB frame
            mock_frame = self._create_mock_visfly_frame(step_count)

            # Use the same frame extraction as evaluation_worker
            import sys
            import importlib.util

            eval_worker_path = Path(__file__).parent.parent.parent / "quadro_llm" / "core" / "evaluation_worker.py"
            spec = importlib.util.spec_from_file_location("evaluation_worker", eval_worker_path)
            eval_worker = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(eval_worker)

            rgb_frame = eval_worker._extract_rgb_frame(mock_frame)

            if rgb_frame is not None and cv2 is not None:
                if video_writer is None:
                    height, width = rgb_frame.shape[:2]
                    # Use the FIXED codec from evaluation_worker.py
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    video_path = video_dir / f"episode_{episode_idx:02d}.mp4"
                    video_writer = cv2.VideoWriter(
                        str(video_path), fourcc, video_fps, (width, height)
                    )

                    if not video_writer.isOpened():
                        self.fail("Video writer should open successfully with AVC1 codec")

                if video_writer is not None:
                    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                    video_writer.write(bgr_frame)
                    frames_captured += 1

        # Clean up
        if video_writer is not None:
            video_writer.release()

        # Verify results
        self.assertIsNotNone(video_path, "Video path should be set")
        self.assertTrue(video_path.exists(), "Video file should be created")
        self.assertEqual(frames_captured, max_steps, f"Should capture all {max_steps} frames")
        self.assertGreater(video_path.stat().st_size, 10000,
                          "Video should have substantial content")

        print(f"✓ Pipeline simulation: {frames_captured} frames, {video_path.stat().st_size} bytes")

    def _create_mock_visfly_frame(self, step: int) -> np.ndarray:
        """Create a mock frame that simulates VisFly env.render() output."""
        # Create a frame with gradient pattern (simulating camera view)
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Create a pattern that changes with each step
        for y in range(height):
            for x in range(width):
                # Create a moving pattern
                r = int(255 * ((x + step * 5) % width) / width)
                g = int(255 * ((y + step * 3) % height) / height)
                b = int(128 + 127 * np.sin(step * 0.1))
                frame[y, x] = [r, g, b]

        return frame

    def test_video_codec_comparison(self):
        """Compare AV01 (old, problematic) vs AVC1 (new, fixed) codecs."""
        if cv2 is None:
            self.skipTest("OpenCV not available")

        codecs_to_test = [
            ("av01", "AV1 (old problematic codec)"),
            ("avc1", "AVC1 (new fixed codec)"),
        ]

        test_results = {}

        for codec_str, codec_desc in codecs_to_test:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec_str)
                video_path = self.temp_dir / f"test_{codec_str}.mp4"

                writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

                if writer.isOpened():
                    # Write test frames
                    for i in range(5):
                        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                        writer.write(frame)
                    writer.release()

                    # Check if file was created with content
                    success = video_path.exists() and video_path.stat().st_size > 1000
                    test_results[codec_str] = {
                        "success": success,
                        "size": video_path.stat().st_size if video_path.exists() else 0
                    }
                else:
                    test_results[codec_str] = {"success": False, "size": 0}
                    writer.release()

            except Exception as e:
                test_results[codec_str] = {"success": False, "error": str(e), "size": 0}

        # Report results
        print("\\nCodec Comparison Results:")
        for codec_str, codec_desc in codecs_to_test:
            result = test_results[codec_str]
            status = "✓" if result["success"] else "✗"
            print(f"{status} {codec_desc}: {result}")

        # AVC1 should work (our fix)
        self.assertTrue(test_results["avc1"]["success"],
                       "AVC1 codec should work after fix")

        # Print recommendation
        if not test_results["av01"]["success"] and test_results["avc1"]["success"]:
            print("\\n🎯 Fix Confirmed: AV01 problematic, AVC1 working correctly!")


if __name__ == "__main__":
    unittest.main(verbosity=2)