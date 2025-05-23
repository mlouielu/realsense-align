import json
import pathlib

import cv2
import click
import numpy as np
import matplotlib.pyplot as plt
import zstandard
from matplotlib.widgets import Slider

import realsense_align as ra


def colors(cap_dir, hw_dir="realsense"):
    hw_dir = cap_dir / hw_dir

    color_video = hw_dir / "color.avi"
    cap = cv2.VideoCapture(str(color_video))
    color_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        color_frames.append(frame)
    cap.release()
    return color_frames


def depths(cap_dir, colors, hw_dir="realsense"):
    hw_dir = cap_dir / hw_dir

    # Prepare configs
    with open(hw_dir / "depth_config.json") as depth_file:
        depth_config = json.load(depth_file)
    depth_intr = depth_config["intrinsics"]
    with open(hw_dir / "color_config.json") as color_file:
        color_config = json.load(color_file)
    color_intr = color_config["intrinsics"]

    depth_scale = depth_config["depth_units"]
    ra_depth_intr = ra.Intrinsics(
        depth_intr["model"],
        depth_intr["width"],
        depth_intr["height"],
        depth_intr["fx"],
        depth_intr["fy"],
        depth_intr["ppx"],
        depth_intr["ppy"],
    )

    ra_color_intr = ra.Intrinsics(
        color_intr["model"],
        color_intr["width"],
        color_intr["height"],
        color_intr["fx"],
        color_intr["fy"],
        color_intr["ppx"],
        color_intr["ppy"],
    )

    depth_file = hw_dir / "depth.zst"
    dctx = zstandard.ZstdDecompressor()
    with open(depth_file, "rb") as f:
        r = dctx.stream_reader(f)
        depths = np.frombuffer(r.read(), dtype=np.uint16)
        depths = (
            depths.reshape((-1, ra_depth_intr.height, ra_depth_intr.width))
            * depth_scale
        )

        aligned = [
            ra.align_z_to_other(depth, color, ra_depth_intr, ra_color_intr, 1)
            for depth, color in zip(depths, colors)
        ]

        return depths, aligned


class InteractiveViewer:
    def __init__(self, colors, depths, aligned_depths):
        self.colors = colors
        self.depths = depths
        self.aligned_depths = aligned_depths
        self.num_frames = min(len(colors), len(depths), len(aligned_depths))
        self.current_frame = 40  # Starting frame

        # Create figure with subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle("RealSense Data Viewer", fontsize=16)

        # Initialize image displays
        self.rgb_im = self.axes[0, 0].imshow(self.colors[self.current_frame])
        self.axes[0, 0].set_title("RGB")
        self.axes[0, 0].axis("off")

        self.depth_im = self.axes[0, 1].imshow(
            self.depths[self.current_frame], cmap="turbo"
        )
        self.axes[0, 1].set_title("Depth")
        self.axes[0, 1].axis("off")

        # RGB with depth overlay
        self.rgb_depth_im = self.axes[1, 0].imshow(self.colors[self.current_frame])
        self.depth_overlay_im = self.axes[1, 0].imshow(
            self.depths[self.current_frame], cmap="turbo", alpha=0.5
        )
        self.axes[1, 0].set_title("RGB + Depth Overlay")
        self.axes[1, 0].axis("off")

        # RGB with aligned depth overlay
        self.rgb_aligned_im = self.axes[1, 1].imshow(self.colors[self.current_frame])
        self.aligned_overlay_im = self.axes[1, 1].imshow(
            self.aligned_depths[self.current_frame], cmap="turbo", alpha=0.5
        )
        self.axes[1, 1].set_title("RGB + Aligned Depth Overlay")
        self.axes[1, 1].axis("off")

        # Add colorbar for depth
        cbar_ax = self.fig.add_axes([0.92, 0.15, 0.02, 0.7])
        self.fig.colorbar(self.depth_im, cax=cbar_ax, label="Depth (m)")

        # Create slider
        slider_ax = self.fig.add_axes([0.1, 0.02, 0.8, 0.03])
        self.slider = Slider(
            slider_ax,
            "Frame",
            valmin=0,
            valmax=self.num_frames - 1,
            valinit=self.current_frame,
            valfmt="%d",
        )

        # Connect slider to update function
        self.slider.on_changed(self.update_frame)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, right=0.9)

    def update_frame(self, val):
        frame_idx = int(self.slider.val)

        # Update RGB
        self.rgb_im.set_array(self.colors[frame_idx])

        # Update depth
        self.depth_im.set_array(self.depths[frame_idx])

        # Update RGB + depth overlay
        self.rgb_depth_im.set_array(self.colors[frame_idx])
        self.depth_overlay_im.set_array(self.depths[frame_idx])

        # Update RGB + aligned depth overlay
        self.rgb_aligned_im.set_array(self.colors[frame_idx])
        self.aligned_overlay_im.set_array(self.aligned_depths[frame_idx])

        # Update main title with frame number
        self.fig.suptitle(f"RealSense Data Viewer - Frame {frame_idx}", fontsize=16)

        # Refresh display
        self.fig.canvas.draw()

    def show(self):
        plt.show()


@click.command()
@click.argument("cap_dir", type=click.Path(exists=True))
def main(cap_dir):
    cols = colors(pathlib.Path(cap_dir))
    deps, aligned = depths(pathlib.Path(cap_dir), cols)

    viewer = InteractiveViewer(cols, deps, aligned)
    viewer.show()


if __name__ == "__main__":
    main()
