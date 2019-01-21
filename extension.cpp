#include <torch/torch.h>

#include <vector>


//CUDA forward declarations
at::Tensor group_point_forward_gpu(const at::Tensor&, const at::Tensor&);
at::Tensor group_point_backward_gpu(const at::Tensor&, const at::Tensor&, const int);


at::Tensor group_points_forward(
    const at::Tensor &points,
    const at::Tensor &indices
)
{
  return group_point_forward_gpu(points, indices);
}
    

at::Tensor group_points_backward(
    const at::Tensor &grad_out,
    const at::Tensor &indices,
    const int n
)
{

  return group_point_backward_gpu(grad_out, indices, n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &group_points_forward, "Grouping points forward (CUDA)");
  m.def("backward", &group_points_backward, "Grouping points backward (CUDA)");
}
