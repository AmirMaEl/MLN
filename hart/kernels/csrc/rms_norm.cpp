// Inspired by TRT-LLM.
// Modified by Shang Yang and Haotian Tang.
// @article{lin2024qserve,
//   title={QServe: W4A8KV4 Quantization and System Co-design for Efficient LLM
//   Serving}, author={Lin*, Yujun and Tang*, Haotian and Yang*, Shang and
//   Zhang, Zhekai and Xiao, Guangxuan and Gan, Chuang and Han, Song},
//   journal={arXiv preprint arXiv:2405.04532},
//   year={2024}
// }
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "layernorm/layernorm.h"



// Bind the function to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm, py::arg("out"), py::arg("input"),
    py::arg("weight"), py::arg("epsilon"), py::arg("use_quant") = false,
    "Apply Root Mean Square (RMS) Normalization to the input tensor.");
}
