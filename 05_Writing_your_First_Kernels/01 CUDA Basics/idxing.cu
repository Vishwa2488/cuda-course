#include <stdio.h>

__global__ void kernel(void)
{
	    int block_id = blockIdx.x 
		                        + blockIdx.y * gridDim.x
					                    + blockIdx.z * gridDim.y * gridDim.x;
	        int thread_id = threadIdx.x 
			                    + threadIdx.y * blockDim.x
					                        + threadIdx.z & blockDim.x * blockDim.y;
		    int id = block_id * blockDim.x * blockDim.y * blockDim.z 
			                + thread_id;
		        
		        printf("%3d | block: %d/%d %d/%d %d/%d = %3d/%3d, thread: %d/%d %d/%d %d/%d = %3d/%3d\n", 
					        id, 
						        blockIdx.x, 
							        gridDim.x, 
								        blockIdx.y, 
									        gridDim.y, 
										        blockIdx.z, 
											        gridDim.z, 
												        block_id, 
													        gridDim.x * gridDim.y * gridDim.z, 
														        threadIdx.x,
															        blockDim.x,
																        threadIdx.y,
																	        blockDim.y,
																		        threadIdx.z,
																			        blockDim.z,
																				        thread_id, 
																					        blockDim.x * blockDim.y * blockDim.z
																						        );
}

int main()
{
	    int block[3] = {2, 3, 4};
	        int thread[3] = { 2, 3, 4};

		    dim3 blocksPerGrid(block[0], block[1], block[2]);
		        dim3 threadsPerBlock(thread[0], thread[1], thread[2]);

			    kernel <<<blocksPerGrid, threadsPerBlock>>> ();

			        cudaDeviceSynchronize();
				    return 0;
}
