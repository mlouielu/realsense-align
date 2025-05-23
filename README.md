realsense-align
===============

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![License: BSD 3-Clause-Clear](https://img.shields.io/badge/License-BSD%203--Clause--Clear-green.svg)](https://spdx.org/licenses/BSD-3-Clause-Clear.html)
[![PyPI - Version](https://img.shields.io/pypi/v/realsense-align)](https://pypi.org/project/realsense-align/)

Porting `librealsense` C++ align code to Python C++ extension. Align
depth and color image from `numpy` array without `librealsense` SDK
and `rs:frame` infrastructure.

Prerequisites
-------------

* uv
* OpenMP

Build
-----

```bash
uv build
```

Install
-------

### Install from PyPI

```bash
python -m pip install realsense-align
```

How to use
----------

See [tests/test_align.py](tests/test_align.py) for detail.

Run the test code by `pytest tests`.

Example
-------

### mmwave-capture-std RGB+Depth Video Player

Set up the virtual environment using uv

```bash
uv sync
source .venv/bin/activate
```

Then, run the code to show the RGB+Depth video

```bash
python examples/play_depth_video.py path/to/capture_00001
```

[IPorting
-------

* [x] struct Intrinsic
* [x] Depth to color align
* [ ] Color to depth align

Trade-off
---------

1. Don't care distortion: See
[librealsense#1430](https://github.com/IntelRealSense/librealsense/issues/1430#issuecomment-375945916)
for more information.

1. Don't care extrinsic: Assume the following:

```en
Rotation: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
Translation: [0, 0, 0]
```

See
[src/proc/align.cpp](https://github.com/IntelRealSense/librealsense/blob/8ffb17b027e100c2a14fa21f01f97a1921ec1e1b/src/proc/align.cpp#L169)
for original implementation, and [rs2_extrinsicsStruct
Reference](https://intelrealsense.github.io/librealsense/doxygen/structrs2__extrinsics.html)
for `rs2_extrinsics` structure.

Links
-----

* License: [BSD 3-Clause Clear License](https://github.com/mlouielu/realsense-align/blob/main/LICENSE)

LICENSE
-------

```text
The Clear BSD License

Copyright (c) 2023 Louie Lu <louielu@cs.unc.edu>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted (subject to the limitations in the disclaimer
below) provided that the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```
