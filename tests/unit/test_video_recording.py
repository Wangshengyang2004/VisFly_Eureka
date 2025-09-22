"""
Unit tests for video recording functionality in evaluation worker.

Tests video codec compatibility, frame processing, and video file creation
to identify and resolve video recording issues.
"""

import os
import tempfile
import unittest
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

import pytest


class TestVideoRecording(unittest.TestCase):
    """Test video recording functionality and codec compatibility."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix="video_test_"))
        self.test_video_fps = 30.0
        self.test_resolution = (640, 480)  # width, height

    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.skipif(cv2 is None, reason="OpenCV not available")
    def test_opencv_availability(self):
        """Test that OpenCV is available and get version info."""
        self.assertIsNotNone(cv2, "OpenCV should be available")
        print(f"OpenCV version: {cv2.__version__}")

    def test_codec_compatibility(self):
        """Test different video codecs for compatibility."""
        if cv2 is None:
            self.skipTest("OpenCV not available")

        codecs_to_test = [
            ("av01", "AV1"),           # Current problematic codec
            ("mp4v", "MP4V"),          # Standard MP4
            ("XVID", "XVID"),          # Xvid codec
            ("MJPG", "MJPG"),          # Motion JPEG
            ("H264", "H.264"),         # H.264
            ("X264", "X.264"),         # X.264
        ]

        compatible_codecs = []
        incompatible_codecs = []

        for codec_str, codec_name in codecs_to_test:
            success, video_path = self._test_codec(codec_str, codec_name)
            if success:
                compatible_codecs.append((codec_str, codec_name))
                # Verify file exists and has content
                if video_path and video_path.exists() and video_path.stat().st_size > 0:
                    print(f"✓ {codec_name} ({codec_str}): Compatible - {video_path.stat().st_size} bytes")
                else:
                    print(f"⚠ {codec_name} ({codec_str}): Writer created but no file output")
            else:
                incompatible_codecs.append((codec_str, codec_name))
                print(f"✗ {codec_name} ({codec_str}): Incompatible")

        # Report results
        print(f"\nCompatible codecs: {len(compatible_codecs)}")
        print(f"Incompatible codecs: {len(incompatible_codecs)}")

        # Assert we have at least one working codec
        self.assertGreater(len(compatible_codecs), 0,
                          "At least one video codec should be compatible")

        # Check if AV1 (the problematic one) is in incompatible list
        av01_incompatible = any(codec[0] == "av01" for codec in incompatible_codecs)
        if av01_incompatible:
            print("⚠ AV1 codec is incompatible - this explains the video recording failure")

        return compatible_codecs, incompatible_codecs

    def _test_codec(self, codec_str: str, codec_name: str) -> Tuple[bool, Optional[Path]]:
        """Test a specific codec by creating a short video."""
        try:
            # Create codec fourcc
            fourcc = cv2.VideoWriter_fourcc(*codec_str)

            # Create video file path
            video_path = self.temp_dir / f"test_{codec_str.lower()}.mp4"

            # Try to create video writer
            writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                self.test_video_fps,
                self.test_resolution
            )

            if not writer.isOpened():
                writer.release()
                return False, None

            # Write a few test frames
            for i in range(10):
                # Create a simple test frame (gradient pattern)
                frame = self._create_test_frame(i)
                writer.write(frame)

            writer.release()

            # Check if file was actually created with content
            if video_path.exists() and video_path.stat().st_size > 0:
                return True, video_path
            else:
                return False, video_path

        except Exception as e:
            print(f"Exception testing {codec_name}: {e}")
            return False, None

    def _create_test_frame(self, frame_number: int) -> np.ndarray:
        """Create a test frame with gradient pattern."""
        height, width = self.test_resolution[1], self.test_resolution[0]

        # Create gradient pattern
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Create a moving pattern
        offset = (frame_number * 10) % width
        for x in range(width):
            intensity = int(255 * ((x + offset) % width) / width)
            frame[:, x] = [intensity, 255 - intensity, 128]

        return frame

    def test_frame_extraction_function(self):
        """Test the _extract_rgb_frame function from evaluation_worker."""
        # Import the function
        import sys
        eval_worker_path = Path(__file__).parent.parent.parent / "quadro_llm" / "core" / "evaluation_worker.py"

        # Load the function dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location("evaluation_worker", eval_worker_path)
        eval_worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(eval_worker)

        extract_rgb_frame = eval_worker._extract_rgb_frame

        # Test different frame formats
        test_cases = [
            # Case 1: Standard RGB frame
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),

            # Case 2: Channel-first format (3, H, W)
            np.random.randint(0, 255, (3, 480, 640), dtype=np.uint8),

            # Case 3: Batch format (1, H, W, 3)
            np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8),

            # Case 4: Grayscale (H, W)
            np.random.randint(0, 255, (480, 640), dtype=np.uint8),

            # Case 5: Dictionary format (simulating env.render() output)
            {"rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)},

            # Case 6: List format (vectorized env)
            [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)],

            # Case 7: None (no render output)
            None,
        ]

        for i, test_frame in enumerate(test_cases):
            with self.subTest(case=i):
                result = extract_rgb_frame(test_frame)

                if test_frame is None:
                    self.assertIsNone(result, f"Case {i}: None input should return None")
                else:
                    if i < 6:  # Valid cases
                        self.assertIsNotNone(result, f"Case {i}: Should extract valid frame")
                        self.assertEqual(result.shape[-1], 3, f"Case {i}: Should have 3 channels")
                        self.assertEqual(result.dtype, np.uint8, f"Case {i}: Should be uint8")

                print(f"Case {i}: {'✓' if result is not None else '✗'} - {type(test_frame)} -> {result.shape if result is not None else 'None'}")

    def test_video_recording_simulation(self):
        """Simulate the video recording process from evaluation_worker."""
        if cv2 is None:
            self.skipTest("OpenCV not available")

        # Test with a known working codec first
        compatible_codecs, _ = self.test_codec_compatibility()

        if not compatible_codecs:
            self.skipTest("No compatible video codecs found")

        # Use the first compatible codec
        codec_str, codec_name = compatible_codecs[0]
        print(f"Testing video recording simulation with {codec_name} codec")

        # Simulate video recording
        video_path = self.temp_dir / "simulation_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*codec_str)

        # Simulate evaluation parameters
        eval_episodes = 3
        max_steps = 20
        video_fps = 30.0

        saved_videos = []

        for episode_idx in range(eval_episodes):
            episode_video_path = self.temp_dir / f"episode_{episode_idx:02d}.mp4"

            video_writer = cv2.VideoWriter(
                str(episode_video_path), fourcc, video_fps, self.test_resolution
            )

            if not video_writer.isOpened():
                print(f"Failed to open video writer for episode {episode_idx}")
                continue

            frames_captured = 0

            # Simulate episode steps
            for step in range(max_steps):
                # Create mock frame (simulating env.render())
                frame = self._create_test_frame(step + episode_idx * max_steps)

                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(bgr_frame)
                frames_captured += 1

            video_writer.release()

            if episode_video_path.exists() and episode_video_path.stat().st_size > 0:
                saved_videos.append(str(episode_video_path))
                print(f"✓ Episode {episode_idx}: {frames_captured} frames, {episode_video_path.stat().st_size} bytes")
            else:
                print(f"✗ Episode {episode_idx}: Video file not created or empty")

        # Verify results
        self.assertGreater(len(saved_videos), 0, "At least one video should be created")
        self.assertEqual(len(saved_videos), eval_episodes, "All episodes should have videos")

        print(f"Video recording simulation: {len(saved_videos)}/{eval_episodes} videos created")

        return saved_videos

    def test_video_playback_verification(self):
        """Test that created videos can be read back."""
        if cv2 is None:
            self.skipTest("OpenCV not available")

        # Create a test video first
        try:
            saved_videos = self.test_video_recording_simulation()
        except Exception as e:
            self.skipTest(f"Could not create test video: {e}")

        if not saved_videos:
            self.skipTest("No test videos were created")

        # Test reading the first video
        video_path = saved_videos[0]
        cap = cv2.VideoCapture(video_path)

        self.assertTrue(cap.isOpened(), "Video should be readable")

        # Read video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video properties: {frame_count} frames, {fps} FPS, {width}x{height}")

        self.assertGreater(frame_count, 0, "Video should have frames")
        self.assertEqual((width, height), self.test_resolution, "Video should have correct resolution")

        # Read a few frames to verify content
        frames_read = 0
        while frames_read < min(5, frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            self.assertIsNotNone(frame, "Frame should be readable")
            self.assertEqual(frame.shape[:2], (height, width), "Frame should have correct dimensions")
            frames_read += 1

        cap.release()

        self.assertGreater(frames_read, 0, "Should be able to read frames from video")
        print(f"✓ Successfully read {frames_read} frames from video")


if __name__ == "__main__":
    unittest.main(verbosity=2)