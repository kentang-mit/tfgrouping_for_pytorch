from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='group_points_cuda',
    ext_modules=[
        CUDAExtension('group_points_cuda', [
            'extension.cpp',
            'group_points.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
