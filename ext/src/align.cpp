//
//
// Copyright (c) 2023 Louie Lu <louielu@cs.unc.edu>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted (subject to the limitations in the disclaimer
// below) provided that the following conditions are met:
//
//      * Redistributions of source code must retain the above copyright notice,
//      this list of conditions and the following disclaimer.
//
//      * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//      * Neither the name of the copyright holder nor the names of its
//      contributors may be used to endorse or promote products derived from this
//      software without specific prior written permission.
//
// NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
// THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
// CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
// IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Modifications: Extract necessary codes from `src/proc/align.cpp` and combine
//                wrapped it with pybind11 to provide fast align from Python
//                without librealsense SDK and the `rs::frame` infrastructure.
//
// License: Apache 2.0. See https://github.com/IntelRealSense/librealsense
//          LICENSE file in root directory
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.
//

#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

struct Intrinsics {
    Intrinsics(const std::string &model,
               double width,
               double height,
               double fx,
               double fy,
               double ppx,
               double ppy)
        : model(model),
          width(width),
          height(height),
          fx(fx),
          fy(fy),
          ppx(ppx),
          ppy(ppy)
    {
    }

    std::string model;
    int width, height;
    double fx, fy;
    double ppx, ppy;
};

void rs2_deproject_pixel_to_point(float point[3],
                                  const struct Intrinsics *intrin,
                                  const float pixel[2],
                                  float depth)
{
    float x = (pixel[0] - intrin->ppx) / intrin->fx;
    float y = (pixel[1] - intrin->ppy) / intrin->fy;

    point[0] = depth * x;
    point[1] = depth * y;
    point[2] = depth;
}

void rs2_project_point_to_pixel(float pixel[2],
                                const struct Intrinsics *intrin,
                                const float point[3])
{
    float x = point[0] / point[2];
    float y = point[1] / point[2];

    pixel[0] = x * intrin->fx + intrin->ppx;
    pixel[1] = y * intrin->fy + intrin->ppy;
}

template <class GET_DEPTH, class TRANSFER_PIXEL>
void align_images(const struct Intrinsics &depth_intrin,
                  const struct Intrinsics &other_intrin,
                  GET_DEPTH get_depth,
                  TRANSFER_PIXEL transfer_pixel)
{
    // Iterate over the pixels of the depth image
#pragma omp parallel for schedule(dynamic)
    for (int depth_y = 0; depth_y < depth_intrin.height; ++depth_y) {
        int depth_pixel_index = depth_y * depth_intrin.width;
        for (int depth_x = 0; depth_x < depth_intrin.width;
             ++depth_x, ++depth_pixel_index) {
            // Skip over depth pixels with the value of zero, we have no depth data so we will not write anything into our aligned images
            if (float depth = get_depth(depth_pixel_index)) {
                // Map the top-left corner of the depth pixel onto the other image
                float depth_pixel[2] = { depth_x - 0.5f, depth_y - 0.5f },
                      depth_point[3], other_point[3], other_pixel[2];
                rs2_deproject_pixel_to_point(depth_point, &depth_intrin,
                                             depth_pixel, depth);

                // We don't have translation for the depth camera, so we just use the depth point
                other_point[0] = depth_point[0];
                other_point[1] = depth_point[1];
                other_point[2] = depth_point[2];
                rs2_project_point_to_pixel(other_pixel, &other_intrin,
                                           other_point);

                const int other_x0 = static_cast<int>(other_pixel[0] + 0.5f);
                const int other_y0 = static_cast<int>(other_pixel[1] + 0.5f);

                // Map the bottom-right corner of the depth pixel onto the other image
                depth_pixel[0] = depth_x + 0.5f;
                depth_pixel[1] = depth_y + 0.5f;
                rs2_deproject_pixel_to_point(depth_point, &depth_intrin,
                                             depth_pixel, depth);
                other_point[0] = depth_point[0];
                other_point[1] = depth_point[1];
                other_point[2] = depth_point[2];
                rs2_project_point_to_pixel(other_pixel, &other_intrin,
                                           other_point);
                const int other_x1 = static_cast<int>(other_pixel[0] + 0.5f);
                const int other_y1 = static_cast<int>(other_pixel[1] + 0.5f);

                if (other_x0 < 0 || other_y0 < 0 ||
                    other_x1 >= other_intrin.width ||
                    other_y1 >= other_intrin.height)
                    continue;

                // Transfer between the depth pixels and the pixels inside the rectangle on the other image
                for (int y = other_y0; y <= other_y1; ++y) {
                    for (int x = other_x0; x <= other_x1; ++x) {
                        transfer_pixel(depth_pixel_index,
                                       y * other_intrin.width + x);
                    }
                }
            }
        }
    }
}



py::array_t<uint16_t> align_z_to_other(py::array_t<int16_t> depth,
                                       py::array_t<uint8_t> color,
                                       const struct Intrinsics &depth_intrin,
                                       const struct Intrinsics &color_intrin,
                                       double z_scale)
{
    // XXX: Fixed size!?
    py::array_t<uint16_t> out_z_arr({ 1080, 1920 });
    py::buffer_info out_z_info = out_z_arr.request();
    py::buffer_info depth_info = depth.request();
    py::buffer_info color_info = color.request();


    if (depth_info.ndim != 2 || color_info.ndim != 3) {
        throw std::runtime_error("Number of dimensions must be two");
    }

    out_z_arr[py::make_tuple(py::ellipsis())] = 0;
    auto *z_pixels = reinterpret_cast<const uint16_t *>(depth_info.ptr);
    auto *out_z = (uint16_t *) (out_z_info.ptr);

    align_images(
        depth_intrin, color_intrin,
        [z_pixels, z_scale](int z_pixel_index) {
            return z_scale * z_pixels[z_pixel_index];
        },
        [out_z, z_pixels](int z_pixel_index, int other_pixel_index) {
            out_z[other_pixel_index] =
                out_z[other_pixel_index]
                    ? std::min((int) out_z[other_pixel_index],
                               (int) z_pixels[z_pixel_index])
                    : z_pixels[z_pixel_index];
        });

    return out_z_arr;
}



PYBIND11_MODULE(realsense_align_ext, m)
{
    m.doc() = "Realsense align";
    m.def("align_z_to_other", &align_z_to_other, "Align depth to other stream");
    py::class_<Intrinsics>(m, "Intrinsics")
        .def(py::init<const std::string &, double, double, double, double,
                      double, double>())
        .def_readwrite("model", &Intrinsics::model)
        .def_readwrite("width", &Intrinsics::width)
        .def_readwrite("height", &Intrinsics::height)
        .def_readwrite("fx", &Intrinsics::fx)
        .def_readwrite("fy", &Intrinsics::fy)
        .def_readwrite("ppx", &Intrinsics::ppx)
        .def_readwrite("ppy", &Intrinsics::ppy);
}
