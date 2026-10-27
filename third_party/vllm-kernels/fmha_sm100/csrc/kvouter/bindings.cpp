/*
 * Copyright (c) 2026 Fireworks AI
 * SPDX-License-Identifier: Apache-2.0
 */

#include <torch/extension.h>

#include "cute_sparse_kvouter.h"

namespace fmha_sm100 {

TORCH_LIBRARY(fmha_sm100, m) {
  m.def("sparse_kvouter_init", sparse_kvouter_init);
  m.def("sparse_kvouter_attn", sparse_kvouter_attn);
}

}  // namespace fmha_sm100

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
