import torch
import torch.nn as nn


def loss_fn(x: torch.Tensor, y: torch.Tensor):
    return ((x-y)**2).sum()

class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
    
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                #result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result


# @ti.kernel
# def ti_update_weights(weight : ti.template(), grad : ti.template(), lr : ti.f32):
#     for I in ti.grouped(weight):
#         weight[I] -= lr * grad[I]


# @ti.kernel
# def ti_update_weights(weight : ti.template(),
#                       grad : ti.template(), grad_1st_moments : ti.template(), grad_2nd_moments : ti.template(),
#                       lr : ti.f32, eps : ti.f32):
#     for I in ti.grouped(weight):
#         g = grad[I]
#         if any(g != 0.0):
#             m = beta1 * grad_1st_moments[I] + (1.0 - beta1) * g
#             v = beta2 * grad_2nd_moments[I] + (1.0 - beta2) * g * g
#             grad_1st_moments[I] = m
#             grad_2nd_moments[I] = v
#             m_hat = m / (1.0 - beta1)
#             v_hat = v / (1.0 - beta2)
#             weight[I] -= lr * m_hat / (ti.sqrt(v_hat) + eps)


# @ti.data_oriented
# class MultiResHashEncoding:
#     def __init__(self, batch_size, temp_hit, model_launch, dim=3) -> None:
#         self.temp_hit = temp_hit
#         self.model_launch = model_launch
#         self.dim = dim
#         self.batch_size = batch_size
#         self.input_positions = ti.Vector.field(self.dim, dtype=data_type, shape=(self.batch_size), needs_grad=False)
#         self.grids = []
#         self.grids_1st_moment = []
#         self.grids_2nd_moment = []

#         self.F = 2 # Number of feature dimensions per entry F = 2
#         # self.N_max = N_max
#         self.N_min = 16
#         self.n_tables = 16
#         # self.b = np.exp((np.log(self.N_max) - np.log(self.N_min)) / (self.n_tables - 1)) # Equation (3)
#         self.max_table_size = 2 ** 19
#         self.per_level_scales = 1.3195079565048218

#         print("n_tables", self.n_tables)
#         self.table_sizes = []
#         self.N_l = []
#         self.n_params = 0
#         self.n_features = 0
#         self.max_direct_map_level = 0
#         for i in range(self.n_tables):
#             # resolution = int(np.floor(self.N_min * (self.b ** i))) # Equation (2)
#             resolution = int(np.ceil(self.N_min * np.exp(i*np.log(self.per_level_scales)) - 1.0)) + 1
#             self.N_l.append(resolution)
#             table_size = resolution**self.dim
#             table_size = int(resolution**self.dim) if table_size % 8 == 0 else int((table_size + 8 - 1) / 8) * 8
#             table_size = min(self.max_table_size, table_size)
#             self.table_sizes.append(table_size)
#             if table_size == resolution ** self.dim:
#                 self.max_direct_map_level = i
#                 # table_size = (resolution + 1) ** self.dim
#             print(f"level {i} resolution: {resolution} n_entries: {table_size}")
            
#             self.grids.append(ti.Vector.field(self.F, dtype=data_type, shape=(table_size), needs_grad=True))
#             self.grids_1st_moment.append(ti.Vector.field(self.F, dtype=data_type, shape=(table_size)))
#             self.grids_2nd_moment.append(ti.Vector.field(self.F, dtype=data_type, shape=(table_size)))
#             self.n_features += self.F
#             self.n_params += self.F * table_size
#         self.encoded_positions = ti.field(dtype=data_type, shape=(self.batch_size, self.n_features), needs_grad=True)
#         self.hashes = [1, 2654435761, 805459861]
        
#         print(f"dim {self.dim}, hash table #params: {self.n_params}")

#     @ti.kernel
#     def initialize(self):
#         for l in ti.static(range(self.n_tables)):
#             for I in ti.grouped(self.grids[l]):
#                 self.grids[l][I] = (ti.Vector([ti.random(), ti.random()]) * 2.0 - 1.0) * 1e-4

#     @ti.func
#     def spatial_hash(self, p, level : ti.template()):
#         hash = ti.uint32(0)
#         if ti.static(level <= self.max_direct_map_level):
#             hash = p.z * self.N_l[level] * self.N_l[level] + p.y * self.N_l[level] + p.x
#         else:
#             for axis in ti.static(range(self.dim)):
#                 hash = hash ^ (p[axis] * ti.uint32(self.hashes[axis]))
#             hash = hash % ti.static(self.table_sizes[level])
#         return int(hash)

#     @ti.kernel
#     def encoding2D(self):
#         for i in self.input_positions:
#             p = self.input_positions[i]
#             for l in ti.static(range(self.n_tables)):
#                 uv = p * ti.cast(self.N_l[l], ti.f32)
#                 iuv = ti.cast(ti.floor(uv), ti.i32)
#                 fuv = ti.math.fract(uv)
#                 c00 = self.grids[l][self.spatial_hash(iuv, l)]
#                 c01 = self.grids[l][self.spatial_hash(iuv + ivec2(0, 1), l)]
#                 c10 = self.grids[l][self.spatial_hash(iuv + ivec2(1, 0), l)]
#                 c11 = self.grids[l][self.spatial_hash(iuv + ivec2(1, 1), l)]
#                 c0 = c00 * (1.0 - fuv[0]) + c10 * fuv[0]
#                 c1 = c01 * (1.0 - fuv[0]) + c11 * fuv[0]
#                 c = c0 * (1.0 - fuv[1]) + c1 * fuv[1]
#                 self.encoded_positions[i, l * 2 + 0] = c.x
#                 self.encoded_positions[i, l * 2 + 1] = c.y


#     @ti.kernel
#     def encoding3D(self):
#         # for i in self.input_positions:
#         for i in range(self.model_launch[None]):
#             p = self.input_positions[self.temp_hit[i]] + 0.5
#             for l in ti.static(range(self.n_tables)):
#                 uvz = p * ti.cast(self.N_l[l], data_type)
#                 iuvz = ti.cast(ti.floor(uvz), ti.i32)
#                 fuvz = ti.math.fract(uvz)
#                 c000 = self.grids[l][self.spatial_hash(iuvz, l)]
#                 c001 = self.grids[l][self.spatial_hash(iuvz + ivec3(0, 0, 1), l)]
#                 c010 = self.grids[l][self.spatial_hash(iuvz + ivec3(0, 1, 0), l)]
#                 c011 = self.grids[l][self.spatial_hash(iuvz + ivec3(0, 1, 1), l)]
#                 c100 = self.grids[l][self.spatial_hash(iuvz + ivec3(1, 0, 0), l)]
#                 c101 = self.grids[l][self.spatial_hash(iuvz + ivec3(1, 0, 1), l)]
#                 c110 = self.grids[l][self.spatial_hash(iuvz + ivec3(1, 1, 0), l)]
#                 c111 = self.grids[l][self.spatial_hash(iuvz + ivec3(1, 1, 1), l)]

#                 c00 = c000 * (1.0 - fuvz[0]) + c100 * fuvz[0]
#                 c01 = c001 * (1.0 - fuvz[0]) + c101 * fuvz[0]
#                 c10 = c010 * (1.0 - fuvz[0]) + c110 * fuvz[0]
#                 c11 = c011 * (1.0 - fuvz[0]) + c111 * fuvz[0]

#                 c0 = c00 * (1.0 - fuvz[1]) + c10 * fuvz[1]
#                 c1 = c01 * (1.0 - fuvz[1]) + c11 * fuvz[1]
#                 c = c0 * (1.0 - fuvz[2]) + c1 * fuvz[1]
#                 self.encoded_positions[i, l * 2 + 0] = c.x
#                 self.encoded_positions[i, l * 2 + 1] = c.y


#     def update(self, lr):
#         for i in range(len(self.grids)):
#             g = self.grids[i]
#             g_1st_momemt = self.grids_1st_moment[i]
#             g_2nd_moment = self.grids_2nd_moment[i]
#             ti_update_weights(g, g.grad, g_1st_momemt, g_2nd_moment, lr, 1e-15)

