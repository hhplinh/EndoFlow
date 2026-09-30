/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_CONFIG_H_INCLUDED
#define CUDA_RASTERIZER_CONFIG_H_INCLUDED

#define NUM_CHANNELS 3 // Default 3, RGB
#define NUM_ALL_MAP 5
#define BLOCK_X 16
#define BLOCK_Y 16

#define N_IN_DATA 2
#define N_OUT_DATA 7
#define N_IN_MAP 3
#define N_OUT_MAP 2

/*
N_IN_DATA
0: gs_flow.x
1: gs_flow.y

N_OUT_DATA
0: gs_flow.x
1: gs_flow.y
2: gs_flow.z
3: gs_weight_sum
4: means2D.x
5: means2D.y
6: projection_valid

N_IN_MAP
0: flow.x
1: flow.y
2: flow.z

N_OUT_MAP
0: optical_flow.x
1: optical_flow.y

*/


#endif