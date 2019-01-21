#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// input: points (b,c,n), idx (b,m,nsample)
// output: out (b,c,m,nsample)
__global__ void group_point_gpu(int b, int c, int n, int m, int nsample, const float *points, const int64_t *idx, float *out) {
    int batch_index = blockIdx.x;
    points += c*n*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    //suppose indices (b,m,nsample). j->m, k->nsample, l->c.
    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            //if(k==0){printf("%ld\n", idx[j*nsample+k]);}
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[l*nsample*m+j*nsample+k] = points[ii+n*l];
            }
        }
    }

}

// input: grad_out (b,c,m,nsample), idx (b,m,nsample), 
// output: grad_points (b,c,n)
__global__ void group_point_grad_gpu(int b, int c, int n, int m, int nsample, const float *grad_out, const int64_t *idx, float *grad_points) {
    int batch_index = blockIdx.x;
    idx += m*nsample*batch_index;
    grad_out += m*nsample*c*batch_index;
    grad_points += n*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                atomicAdd(&grad_points[l*n+ii], grad_out[l*m*nsample+j*nsample+k]);
            }
        }
    }
}


at::Tensor group_point_forward_gpu(const at::Tensor& points, const at::Tensor& indices){
    int b = points.size(0);
    int c = points.size(1);
    int n = points.size(2);
    int m = indices.size(1);
    int nsample = indices.size(2);
    //make sure out is a variable.    
    at::Tensor out = torch::zeros(std::vector<int64_t>({b,c,m,nsample}), torch::dtype(torch::kFloat32).requires_grad(true).device(torch::kCUDA, 0));
    group_point_gpu<<<b,256>>>(b,c,n,m,nsample,points.data<float>(),indices.data<int64_t>(),out.data<float>());
    
    cudaDeviceSynchronize();
    return out;
}

at::Tensor group_point_backward_gpu(const at::Tensor& grad_out, const at::Tensor& idx, const int n){
    int b = grad_out.size(0);
    int c = grad_out.size(1);
    int m = grad_out.size(2);
    int nsample = grad_out.size(3);
    at::Tensor grad_points = torch::zeros(std::vector<int64_t>({b, c, n}), torch::dtype(torch::kFloat32).requires_grad(true).device(torch::kCUDA, 0));
    group_point_grad_gpu<<<b,256>>>(b,c,n,m,nsample,grad_out.data<float>(),idx.data<int64_t>(),grad_points.data<float>());
    cudaDeviceSynchronize();
    return grad_points;
}


