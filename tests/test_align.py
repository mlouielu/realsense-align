#
# Copyright (c) 2023 Louie Lu <louielu@cs.unc.edu>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#      * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from this
#      software without specific prior written permission.
#
# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

import pytest

import pathlib

import numpy as np

import realsense_align


@pytest.fixture
def test_data():
    return np.load(
        pathlib.Path(__file__).parent / "data" / "test_data.npz", allow_pickle=True
    )


@pytest.fixture
def depth(test_data):
    return test_data["depth"]


@pytest.fixture
def color(test_data):
    return test_data["color"]


@pytest.fixture
def depth_scale(test_data):
    return test_data["depth_scale"]


@pytest.fixture
def depth_intr(test_data):
    intr = test_data["depth_intr"][()]
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
def color_intr(test_data):
    intr = test_data["color_intr"][()]
    return realsense_align.Intrinsics(
        intr["model"],
        intr["width"],
        intr["height"],
        intr["fx"],
        intr["fy"],
        intr["ppx"],
        intr["ppy"],
    )


def test_align_z_to_other(depth, color, depth_intr, color_intr, depth_scale):
    import cv2
    import matplotlib.pyplot as plt

    aligned = realsense_align.align_z_to_other(
        depth, color, depth_intr, color_intr, depth_scale
    )
    assert aligned is not None

    grey_color = 153
    depth_image_3d = np.dstack((aligned, aligned, aligned))
    clipping_distance = 2 / depth_scale
    bg_removed = np.where(
        (depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color
    )

    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(aligned, alpha=0.03), cv2.COLORMAP_JET
    )

    bg_removed = np.rot90(bg_removed, k=-1)
    depth_colormap = np.rot90(depth_colormap, k=-1)

    images = np.hstack((bg_removed, depth_colormap))
    # plt.imshow(images)
    # plt.show()
