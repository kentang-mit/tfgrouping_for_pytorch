from torch.utils.cpp_extension import load
lltm_cuda = load(
    'group_points_cuda', ['extension.cpp', 'group_points.cu'], verbose=True)
help(lltm_cuda)
