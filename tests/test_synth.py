import pytest
import pathlib
import numpy as np
import realsense_align
from typing import Dict, Any
from dataclasses import dataclass

# Test data configuration
TEST_DATA_DIR = pathlib.Path(__file__).parent / "data"


@dataclass
class TestDataConfig:
    """Configuration for different test scenarios"""

    width: int = 640
    height: int = 480
    fx: float = 380.918
    fy: float = 380.918
    ppx: float = 315.489
    ppy: float = 237.560
    depth_scale: float = 0.001
    model: str = "brown_conrady"


class TestDataFactory:
    """Factory for creating controlled test data"""

    @staticmethod
    def create_synthetic_depth(width: int = 640, height: int = 480) -> np.ndarray:
        """Create synthetic depth data with known patterns"""
        # Create gradient depth with some objects
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        depth = np.ones((height, width), dtype=np.uint16) * 1000  # 1m background

        # Add some "objects" at different depths
        depth[100:200, 200:300] = 500  # 50cm object
        depth[300:400, 400:500] = 1500  # 1.5m object
        depth[50:150, 50:150] = 300  # 30cm close object

        return depth

    @staticmethod
    def create_synthetic_color(width: int = 640, height: int = 480) -> np.ndarray:
        """Create synthetic color image with recognizable patterns"""
        color = np.zeros((height, width, 3), dtype=np.uint8)

        # Create colored regions matching depth objects
        color[100:200, 200:300] = [255, 0, 0]  # Red object
        color[300:400, 400:500] = [0, 255, 0]  # Green object
        color[50:150, 50:150] = [0, 0, 255]  # Blue object
        color[:, :] += 50  # Add some base brightness

        return color

    @staticmethod
    def create_intrinsics(config: TestDataConfig) -> realsense_align.Intrinsics:
        """Create camera intrinsics from config"""
        return realsense_align.Intrinsics(
            config.model,
            config.width,
            config.height,
            config.fx,
            config.fy,
            config.ppx,
            config.ppy,
        )


# Fixtures for real test data
@pytest.fixture
def real_test_data():
    """Load real sensor data from file"""
    data_path = TEST_DATA_DIR / "test_data.npz"
    if not data_path.exists():
        pytest.skip(f"Real test data not found at {data_path}")
    return np.load(data_path, allow_pickle=True)


@pytest.fixture
def real_depth(real_test_data):
    return real_test_data["depth"]


@pytest.fixture
def real_color(real_test_data):
    return real_test_data["color"]


@pytest.fixture
def real_depth_scale(real_test_data):
    return real_test_data["depth_scale"]


@pytest.fixture
def real_depth_intr(real_test_data):
    intr = real_test_data["depth_intr"][()]
    return realsense_align.Intrinsics(
        intr["model"],
        intr["width"],
        intr["height"],
        intr["fx"],
        intr["fy"],
        intr["ppx"],
        intr["ppy"],
    )


@pytest.fixture
def real_color_intr(real_test_data):
    intr = real_test_data["color_intr"][()]
    return realsense_align.Intrinsics(
        intr["model"],
        intr["width"],
        intr["height"],
        intr["fx"],
        intr["fy"],
        intr["ppx"],
        intr["ppy"],
    )


# Fixtures for synthetic test data
@pytest.fixture
def test_config():
    return TestDataConfig()


@pytest.fixture
def synthetic_depth(test_config):
    return TestDataFactory.create_synthetic_depth(test_config.width, test_config.height)


@pytest.fixture
def synthetic_color(test_config):
    return TestDataFactory.create_synthetic_color(test_config.width, test_config.height)


@pytest.fixture
def synthetic_depth_intr(test_config):
    return TestDataFactory.create_intrinsics(test_config)


@pytest.fixture
def synthetic_color_intr(test_config):
    # Slightly different intrinsics for color camera
    config = TestDataConfig(fx=610.0, fy=610.0, ppx=325.0, ppy=245.0)
    return TestDataFactory.create_intrinsics(config)


@pytest.fixture
def synthetic_depth_scale(test_config):
    return test_config.depth_scale


# Parametrized fixtures for different test scenarios
@pytest.fixture(params=["synthetic", "real"])
def test_data_set(
    request,
    synthetic_depth,
    synthetic_color,
    synthetic_depth_intr,
    synthetic_color_intr,
    synthetic_depth_scale,
    real_depth,
    real_color,
    real_depth_intr,
    real_color_intr,
    real_depth_scale,
):
    """Parametrized fixture to test with both synthetic and real data"""
    if request.param == "synthetic":
        return {
            "depth": synthetic_depth,
            "color": synthetic_color,
            "depth_intr": synthetic_depth_intr,
            "color_intr": synthetic_color_intr,
            "depth_scale": synthetic_depth_scale,
        }
    else:
        return {
            "depth": real_depth,
            "color": real_color,
            "depth_intr": real_depth_intr,
            "color_intr": real_color_intr,
            "depth_scale": real_depth_scale,
        }


def test_specific_expected_values(
    synthetic_depth,
    synthetic_color,
    synthetic_depth_intr,
    synthetic_color_intr,
    synthetic_depth_scale,
):
    """Test specific pixel values after alignment"""
    aligned = realsense_align.align_z_to_other(
        synthetic_depth,
        synthetic_color,
        synthetic_depth_intr,
        synthetic_color_intr,
        synthetic_depth_scale,
    )

    # Test exact expected values (with small tolerance for numerical precision)
    assert (
        abs(aligned[400, 500] - 1500) <= 5
    ), f"Expected ~1500mm at [400,500], got {aligned[500, 400]}"
    assert (
        abs(aligned[50, 50] - 300) <= 5
    ), f"Expected ~1000mm at [50,50], got {aligned[50, 50]}"


# Enhanced tests with better assertions
def test_align_z_to_other_basic_functionality(test_data_set):
    """Test basic alignment functionality"""
    data = test_data_set

    aligned = realsense_align.align_z_to_other(
        data["depth"],
        data["color"],
        data["depth_intr"],
        data["color_intr"],
        data["depth_scale"],
    )

    # Basic assertions
    assert aligned is not None, "Alignment should not return None"
    assert isinstance(aligned, np.ndarray), "Result should be numpy array"
    assert (
        aligned.shape == data["color"].shape[:2]
    ), "Aligned depth should match color dimensions"
    assert aligned.dtype == np.uint16


def test_align_z_to_other_value_ranges(
    synthetic_depth,
    synthetic_color,
    synthetic_depth_intr,
    synthetic_color_intr,
    synthetic_depth_scale,
):
    """Test that aligned values are within expected ranges"""
    aligned = realsense_align.align_z_to_other(
        synthetic_depth,
        synthetic_color,
        synthetic_depth_intr,
        synthetic_color_intr,
        synthetic_depth_scale,
    )

    # Value range assertions
    assert np.all(aligned >= 0), "Depth values should be non-negative"
    assert np.any(aligned > 0), "Should have some non-zero depth values"

    # Check that synthetic objects are still present in some form
    non_zero_pixels = np.count_nonzero(aligned)
    total_pixels = aligned.shape[0] * aligned.shape[1]
    assert (
        non_zero_pixels / total_pixels > 0.1
    ), "Should have reasonable coverage of depth data"


def test_align_edge_cases():
    """Test edge cases with minimal/extreme data"""
    config = TestDataConfig(width=100, height=100)

    # Test with zero depth
    zero_depth = np.zeros((100, 100), dtype=np.uint16)
    color = TestDataFactory.create_synthetic_color(100, 100)
    intr = TestDataFactory.create_intrinsics(config)

    aligned = realsense_align.align_z_to_other(
        zero_depth, color, intr, intr, config.depth_scale
    )

    assert aligned is not None, "Should handle zero depth gracefully"
    assert np.all(aligned == 0), "Zero depth should remain zero"


@pytest.mark.parametrize("width,height", [(320, 240), (640, 480), (1280, 720)])
def test_align_different_resolutions(width, height):
    """Test alignment with different image resolutions"""
    config = TestDataConfig(width=width, height=height, ppx=width / 2, ppy=height / 2)

    depth = TestDataFactory.create_synthetic_depth(width, height)
    color = TestDataFactory.create_synthetic_color(width, height)
    intr = TestDataFactory.create_intrinsics(config)

    aligned = realsense_align.align_z_to_other(
        depth, color, intr, intr, config.depth_scale
    )

    assert aligned is not None, f"Should handle {width}x{height} resolution"
    assert aligned.shape == (
        height,
        width,
    ), f"Output shape should match input {height}x{width}"


# Utility function to save test data (for creating new test datasets)
def create_test_data_file():
    """Utility to create/update test data file with synthetic data"""
    config = TestDataConfig()

    test_data = {
        "depth": TestDataFactory.create_synthetic_depth(),
        "color": TestDataFactory.create_synthetic_color(),
        "depth_scale": config.depth_scale,
        "depth_intr": {
            "model": config.model,
            "width": config.width,
            "height": config.height,
            "fx": config.fx,
            "fy": config.fy,
            "ppx": config.ppx,
            "ppy": config.ppy,
        },
        "color_intr": {
            "model": config.model,
            "width": config.width,
            "height": config.height,
            "fx": config.fx + 10,  # Slightly different
            "fy": config.fy + 10,
            "ppx": config.ppx + 5,
            "ppy": config.ppy + 5,
        },
    }

    output_path = TEST_DATA_DIR / "synthetic_test_data.npz"
    output_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(output_path, **test_data)
    print(f"Created synthetic test data at {output_path}")


if __name__ == "__main__":
    create_test_data_file()
