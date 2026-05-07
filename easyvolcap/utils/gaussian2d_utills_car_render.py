import os
import math
import numpy as np
from plyfile import PlyData, PlyElement

import torch
from torch import nn
from torch.optim import Optimizer
from torch.nn import functional as F

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.sh_utils import eval_sh
from easyvolcap.utils.net_utils import make_buffer, make_params, typed
from easyvolcap.utils.light import EnvLight, EnvLightMip



def fov2focal(fov, pixels):
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * np.arctan(pixels / (2 * focal))


@torch.jit.script
def getWorld2View(R: torch.Tensor, t: torch.Tensor):
    """
    R: ..., 3, 3
    T: ..., 3, 1
    """
    sh = R.shape[:-2]
    T = torch.eye(4, dtype=R.dtype, device=R.device)  # 4, 4
    for i in range(len(sh)):
        T = T.unsqueeze(0)
    T = T.expand(sh + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3:] = t
    return T


@torch.jit.script
def getProjectionMatrix(fovx: torch.Tensor, fovy: torch.Tensor, znear: torch.Tensor, zfar: torch.Tensor):
    tanfovy = math.tan((fovy / 2))
    tanfovx = math.tan((fovx / 2))

    t = tanfovy * znear
    b = -t
    r = tanfovx * znear
    l = -r

    P = torch.zeros(4, 4, dtype=znear.dtype, device=znear.device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (r - l)
    P[1, 1] = 2.0 * znear / (t - b)

    P[0, 2] = (r + l) / (r - l)
    P[1, 2] = (t + b) / (t - b)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)

    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P


def prepare_gaussian_camera(batch):
    output = dotdict()
    H, W, K, R, T, n, f = batch.H[0], batch.W[0], batch.K[0], batch.R[0], batch.T[0], batch.n[0], batch.f[0]
    cpu_H, cpu_W, cpu_K, cpu_R, cpu_T, cpu_n, cpu_f = batch.meta.H[0], batch.meta.W[0], batch.meta.K[0], batch.meta.R[0], batch.meta.T[0], batch.meta.n[0], batch.meta.f[0]

    output.image_height = cpu_H
    output.image_width = cpu_W

    output.K = K
    output.R = R
    output.T = T

    fl_x = cpu_K[0, 0]  # use cpu K
    fl_y = cpu_K[1, 1]  # use cpu K
    FoVx = focal2fov(fl_x, cpu_W)
    FoVy = focal2fov(fl_y, cpu_H)
    if 'cmsk' not in batch:
        meta = batch.meta if 'meta' in batch else dotdict()

        def safe_get(d, k):
            try:
                return d[k]
            except Exception:
                return None

        raise RuntimeError(
            "batch has no cmsk. "
            f"available batch keys={list(batch.keys())}, "
            f"meta keys={list(meta.keys()) if hasattr(meta, 'keys') else None}, "
            f"iter={safe_get(meta, 'iter')}, "
            f"view_index={safe_get(meta, 'view_index')}, "
            f"latent_index={safe_get(meta, 'latent_index')}, "
            f"camera_index={safe_get(meta, 'camera_index')}, "
            f"frame_index={safe_get(meta, 'frame_index')}, "
            f"H={safe_get(meta, 'H')}, "
            f"W={safe_get(meta, 'W')}"
        )

    car_mask = batch.cmsk[0]
    output.car_mask = (car_mask > 0.5).float()


    assert output.car_mask.numel() == H * W, \
        f"car_mask numel mismatch: shape={output.car_mask.shape}, numel={output.car_mask.numel()}, expected={H * W}, H={H}, W={W}"

    output.world_view_transform = getWorld2View(R, T).transpose(0, 1)
    output.projection_matrix = getProjectionMatrix(FoVx, FoVy, n, f).transpose(0, 1)
    output.full_proj_transform = torch.matmul(output.world_view_transform, output.projection_matrix)
    output.camera_center = (-R.mT @ T)[..., 0]  # B, 3, 1 -> 3,

    # Set up rasterization configuration
    output.FoVx = FoVx
    output.FoVy = FoVy
    output.tanfovx = math.tan(FoVx * 0.5)
    output.tanfovy = math.tan(FoVy * 0.5)

    output.znear = n
    output.zfar = f

    return output


@torch.jit.script
def rgb2sh0(rgb: torch.Tensor):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


@torch.jit.script
def sh02rgb(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


@torch.jit.script
def scaling_activation(x):
    return torch.exp(x)


@torch.jit.script
def scaling_inverse_activation(x):
    return torch.log(x.clamp(1e-6, 1e6))


@torch.jit.script
def opacity_activation(x):
    return torch.sigmoid(x)

@torch.jit.script
def neighbor_activation(x):
    return torch.sigmoid(x)

@torch.jit.script
def inverse_neighbor_activation(x):
    return torch.logit(torch.clamp(x, 1e-6, 1 - 1e-6))

@torch.jit.script
def inverse_opacity_activation(x):
    return torch.logit(torch.clamp(x, 1e-6, 1 - 1e-6))


@torch.jit.script
def specular_activation(x):
    return torch.sigmoid(x)


@torch.jit.script
def inverse_specular_activation(x):
    return torch.logit(torch.clamp(x, 1e-6, 1 - 1e-6))


def build_rotation(r):
    """ Build a rotation matrix from a quaternion, the
        default quaternion convention is (w, x, y, z).

    Args:
        r (torch.Tensor), (..., 4): the quaternion.
    
    Returns:
        R (torch.Tensor), (..., 3, 3): the rotation matrix
    """

    # Normalize the quaternion
    s = torch.norm(r, dim=-1)[..., None]  # (..., 1)
    q = r / s  # (..., 4)

    # Extract the quaternion components in (w, x, y, z) order
    r = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    # Build the rotation matrix, column-major
    R = torch.zeros(q.shape[:-1] + (3, 3), dtype=r.dtype, device=r.device)  # (..., 3, 3)
    R[..., 0, 0] = 1 - 2 * (y*y + z*z)
    R[..., 0, 1] = 2 * (x*y - r*z)
    R[..., 0, 2] = 2 * (x*z + r*y)
    R[..., 1, 0] = 2 * (x*y + r*z)
    R[..., 1, 1] = 1 - 2 * (x*x + z*z)
    R[..., 1, 2] = 2 * (y*z - r*x)
    R[..., 2, 0] = 2 * (x*z - r*y)
    R[..., 2, 1] = 2 * (y*z + r*x)
    R[..., 2, 2] = 1 - 2 * (x*x + y*y)

    return R


def build_scaling_rotation(s, r):
    L = torch.zeros(s.shape[:-1] + (3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)

    L[..., 0, 0] = s[..., 0]
    L[..., 1, 1] = s[..., 1]
    L[..., 2, 2] = s[..., 2]

    L = R @ L
    return L


@torch.jit.script
def build_cov(center: torch.Tensor, s: torch.Tensor, scaling_modifier: float, q: torch.Tensor):
    L = build_scaling_rotation(torch.cat([s * scaling_modifier, torch.ones_like(s)], dim=-1), q).permute(0, 2, 1)
    T = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device=L.device)
    T[:, :3, :3] = L
    T[:,  3, :3] = center
    T[:,  3,  3] = 1
    return T


def build_inverse_scaling_rotation(s, r):
    L = torch.zeros(s.shape[:-1] + (3, 3), dtype=s.dtype, device=s.device)
    R = build_rotation(r)

    L[..., 0, 0] = 1 / s[..., 0]
    L[..., 1, 1] = 1 / s[..., 1]
    L[..., 2, 2] = 1 / s[..., 2]

    L = L @ R.mT
    return L


@torch.jit.script
def build_inv_cov(center: torch.Tensor, s: torch.Tensor, scaling_modifier: float, q: torch.Tensor):
    L = build_inverse_scaling_rotation(torch.cat([s * scaling_modifier, torch.ones_like(s)], dim=-1), q)  # S^{-1} @ R^T
    T = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device=L.device)  # (P, 4, 4)
    T[:, :3, :3] = L  # (P, 3, 3)
    T[:, :3, 3:] = -L @ center[..., None]  # (P, 3, 1)
    T[:,  3,  3] = 1
    return T


def get_expon_lr_func(
    lr_init,
    lr_final,
    lr_delay_steps=0,
    lr_delay_mult=1.0,
    max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def quat_to_normal_z(rots: torch.Tensor) -> torch.Tensor:
    rots = F.normalize(rots, dim=-1)
    w, x, y, z = rots.unbind(dim=-1)

    normals = torch.stack([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y)
    ], dim=-1)

    return F.normalize(normals, dim=-1)

def get_env_direction(H, W): # give world view(x,y,z)  env_map direction
    gy, gx = torch.meshgrid(torch.linspace(0.0 + 1.0 / H, 1.0 - 1.0 / H, H, device='cuda'), 
                            torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device='cuda'),
                            indexing='ij')
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    env_directions = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    return env_directions

class GaussianModel(nn.Module):
    def __init__(
        self,
        xyz: torch.Tensor = None,
        colors: torch.Tensor = None,
        init_occ: float = 0.1,
        init_scale: torch.Tensor = None,
        sh_degree: int = 3,
        init_sh_degree: int = 0,
        spatial_scale: float = 1.0,
        xyz_lr_scheduler: dotdict = dotdict(max_steps=30000),
        # Reflection related parameters
        render_reflection: bool = False,
        specular_channels: int = 1,
        init_specular: float = 1e-3,
        init_roughness: float = 0.5,
        max_gs: int = 1e6,
        max_gs_threshold: float = 0.9,
        neighbor_effect: float =0.9
        
    ):
        super().__init__()

        self.setup_functions(
            scaling_activation=scaling_activation,
            scaling_inverse_activation=scaling_inverse_activation,
            opacity_activation=opacity_activation,
            inverse_opacity_activation=inverse_opacity_activation
        )

        # SH realte configs
        self.active_sh_degree = make_buffer(torch.full((1,), init_sh_degree, dtype=torch.long))  # save them, but need to keep a copy on cpu
        self.cpu_active_sh_degree = self.active_sh_degree.item() 
        self.max_sh_degree = sh_degree

        # Set scene spatial scale
        self.spatial_scale = spatial_scale


        # EnvLight Settings
        self.envmap_resolution = 128
        self.envmap_max_roughness = 0.5
        self.envmap_min_roughness = 0.08
        self.env_map = None
        self.env_H, self.env_W = 256, 512
        self.env_directions = get_env_direction(self.env_H, self.env_W)

        # Initalize trainable parameters
        self.create_from_pcd(xyz, colors, init_occ, init_scale, specular_channels, init_specular, init_roughness,neighbor_effect)
        self.render_reflection = render_reflection
        self.specular_channels = specular_channels
        self.init_specular = init_specular
        self.init_roughness = init_roughness

        # Densification related parameters
        self.max_radii2D = make_buffer(torch.zeros(self.get_xyz.shape[0]))
        self.xyz_gradient_accum = make_buffer(torch.zeros((self.get_xyz.shape[0], 1)))
        self.denom = make_buffer(torch.zeros((self.get_xyz.shape[0], 1)))
        self.xyz_weight_accum = make_buffer(torch.zeros((self.get_xyz.shape[0], 1)))  # (P, 1)

        self.render_xyz = None

        self.render_features_dc = None
        self.render_features_rest = None
        self.render_opacity = None
        self.render_scaling = None
        self.render_rotation = None

        self.render_specular = None
        self.render_roughness = None
        self.render_neighbor_effect = None
        self.render_car_gaussian_mask = None

        self.max_gs = max_gs
        self.max_gs_threshold = max_gs_threshold

        if xyz_lr_scheduler is not None:
            xyz_lr_scheduler['lr_init'] *= self.spatial_scale
            xyz_lr_scheduler['lr_final'] *= self.spatial_scale
            self.xyz_scheduler = get_expon_lr_func(**xyz_lr_scheduler)
            log(magenta(f'[INIT] Using xyz learning rate scheduler, lr_init: {xyz_lr_scheduler["lr_init"]}, lr_final: {xyz_lr_scheduler["lr_final"]}'))
        else:
            self.xyz_scheduler = None

        # Perform some model messaging before loading
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)
        self.post_handle = self.register_load_state_dict_post_hook(self._load_state_dict_post_hook)

    def setup_functions(
        self,
        scaling_activation=torch.exp,
        scaling_inverse_activation=torch.log,
        opacity_activation=torch.sigmoid,
        inverse_opacity_activation=torch.logit,
        rotation_activation=F.normalize,
        specular_activation=torch.sigmoid,
        specular_inverse_activation=torch.logit,
        roughness_activation=torch.sigmoid,
        roughness_inverse_activation=torch.logit,
        neighbor_activation=torch.sigmoid,
        neighbor_inverse_activation=torch.logit
    ):
        self.scaling_activation = getattr(torch, scaling_activation) if isinstance(scaling_activation, str) else scaling_activation
        self.opacity_activation = getattr(torch, opacity_activation) if isinstance(opacity_activation, str) else opacity_activation
        self.rotation_activation = getattr(torch, rotation_activation) if isinstance(rotation_activation, str) else rotation_activation
        self.scaling_inverse_activation = getattr(torch, scaling_inverse_activation) if isinstance(scaling_inverse_activation, str) else scaling_inverse_activation
        self.opacity_inverse_activation = getattr(torch, inverse_opacity_activation) if isinstance(inverse_opacity_activation, str) else inverse_opacity_activation
        self.covariance_activation = build_cov
        self.inverse_covariance_activation = build_inv_cov

        self.specular_activation = getattr(torch, specular_activation) if isinstance(specular_activation, str) else specular_activation
        self.specular_inverse_activation = getattr(torch, specular_inverse_activation) if isinstance(specular_inverse_activation, str) else specular_inverse_activation
        self.roughness_activation = getattr(torch, roughness_activation) if isinstance(roughness_activation, str) else roughness_activation
        self.roughness_inverse_activation = getattr(torch, roughness_inverse_activation) if isinstance(roughness_inverse_activation, str) else roughness_inverse_activation
        self.neighbor_activation = getattr(torch, neighbor_activation) if isinstance(neighbor_activation, str) else neighbor_activation
        self.neighbor_inverse_activation = getattr(torch, neighbor_inverse_activation) if isinstance(neighbor_inverse_activation, str) else neighbor_inverse_activation

    def render_env_map(self, H=512):
        if H == self.env_H:
            directions = self.env_directions
        else:
            W = H * 2
            directions = get_env_direction(H, W)
        return  self.env_map(directions, mode="pure_env")
    @property   
    def get_envmap(self): 
        return self.env_map

    @property
    def device(self):
        return self.get_xyz.device

    @property
    def number(self):
        return self._xyz.shape[0]

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz
    @property
    def get_neighbor_effect(self):
        return self.neighbor_activation(self._neighbor_effect)

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_specular(self):
        return self.specular_activation(self._specular)

    @property
    def get_roughness(self):
        return self.roughness_activation(self._roughness)

    @property
    def get_max_sh_channels(self):
        return (self.max_sh_degree + 1)**2
    
    @property
    def get_render_scaling(self):
        return self.scaling_activation(self.render_scaling)
    @property
    def get_render_xyz(self):
        return self.render_xyz

    @property
    def get_render_rotation(self):
        return self.rotation_activation(self.render_rotation)

    @property
    def get_render_features(self):
        features_dc = self.render_features_dc
        features_rest = self.render_features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_render_opacity(self):
        return self.opacity_activation(self.render_opacity)
 
    @property
    def get_render_specular(self):
        return self.specular_activation(self.render_specular)
    @property
    def get_render_neighbor_effect(self):
        return self.neighbor_activation(self.render_neighbor_effect)


    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self.get_rotation)

    def get_inverse_covariance(self, scaling_modifier=1):
        return self.inverse_covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self.get_rotation)

    def oneupSHdegree(self):
        changed = False
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            self.cpu_active_sh_degree = self.active_sh_degree.item()
            changed = True
        return changed

    def create_from_pcd(
        self,
        xyz: torch.Tensor,
        colors: torch.Tensor = None,
        opacities: float = 0.1,
        scales: torch.Tensor = None,
        specular_channels: int = 1,
        specular: float = 1e-3,
        roughness: float = 0.5,
        neighbor_effect: float=0.9,
        car_gaussian_mask: float =0.0
    ):
        if xyz is None:
            xyz = torch.empty(1, 3, device='cuda')  # by default, init empty gaussian model on CUDA

        features = torch.zeros((xyz.shape[0], 3, self.get_max_sh_channels))
        if colors is not None:
            features[:, :3, 0] = rgb2sh0(colors)
        features[:, 3: 1:] = 0.0

        log(magenta(f'[INIT] NUM POINTS: {xyz.shape[0]}'))

        if len(xyz) > 1:  # 1 means we're doing a noop initialization
            if scales is None:
                from simple_knn._C import distCUDA2
                dist2 = torch.clamp_min(distCUDA2(xyz.float().cuda()), 0.0000001).cpu()
                scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 2)  # NOTE: 2DGS has only 2 scaling parameters
            else:
                should_recompute = (scales == -1).any(-1)
                if should_recompute.any():
                    from simple_knn._C import distCUDA2
                    dist2 = torch.clamp_min(distCUDA2(xyz[should_recompute].float().cuda()), 0.0000001).cpu()
                    recompute = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(1, 2)  # NOTE: 2DGS has only 2 scaling parameters
                    scales[should_recompute] = recompute  # -1 for computed init
        elif scales is None:
            scales = torch.empty(1, 2)  # NOTE: 2DGS has only 2 scaling parameters
        else:
            scales = self.scaling_inverse_activation(scales)

        rots = torch.rand((xyz.shape[0], 4))

        if not isinstance(opacities, torch.Tensor) or len(opacities) != len(xyz):
            opacities = opacities * torch.ones((xyz.shape[0], 1), dtype=torch.float)
        opacities = self.opacity_inverse_activation(opacities)

        if not isinstance(neighbor_effect , torch.Tensor) or len(neighbor_effect) != len(xyz):
            neighbor_effect= neighbor_effect * torch.ones((xyz.shape[0], 1), dtype=torch.float)
#       neighbor_effect= self.neighbor_inverse_activation(neighbor_effect)

        car_gaussian_mask = car_gaussian_mask * torch.ones( (xyz.shape[0], 1), dtype=torch.float)


        self._xyz = make_params(xyz)
        self._features_dc = make_params(features[:, :, :1].transpose(1, 2).contiguous())
        self._features_rest = make_params(features[:, :, 1:].transpose(1, 2).contiguous())
        self._scaling = make_params(scales)
        self._rotation = make_params(rots)
        self._opacity = make_params(opacities)
        self._neighbor_effect= make_params(neighbor_effect)

        self.env_map= EnvLightMip(path=None, device='cuda', max_res=self.envmap_resolution, min_roughness=self.envmap_min_roughness, max_roughness=self.envmap_max_roughness).cuda()
        self.car_gaussian_mask=make_buffer(car_gaussian_mask)

        if not isinstance(specular, torch.Tensor) or len(specular) != len(xyz):
            specular = specular * torch.ones((xyz.shape[0], specular_channels), dtype=torch.float)
        specular = self.specular_inverse_activation(specular)
        if not isinstance(roughness, torch.Tensor) or len(roughness) != len(xyz):
            roughness = roughness * torch.ones((xyz.shape[0], 1), dtype=torch.float)
        roughness = self.roughness_inverse_activation(roughness)
        self._specular = make_params(specular)
        self._roughness = make_params(roughness)

    @torch.no_grad()
    def _load_state_dict_pre_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Supports loading points and features with different shapes
        if prefix != '' and not prefix.endswith('.'): prefix = prefix + '.'  # special care for when we're loading the model directly
        for name, params in self.named_parameters():
            if f'{prefix}{name}' in state_dict:
                params.data = params.data.new_empty(state_dict[f'{prefix}{name}'].shape)

    @torch.no_grad()
    def _load_state_dict_post_hook(self, module, incompatible_keys):
        # TODO: make this a property that updates the cpu copy on change
        self.cpu_active_sh_degree = self.active_sh_degree.item()

    def distort_color(self, range: float = 0.4, threshold: float = 0.05, optimizer: Optimizer = None, prefix: str = ''):
        log(yellow_slim(f'[DISTORT COLOR]'))
        mask = self.get_specular.max(dim=-1).values.flatten() > threshold
        features_dc = self._features_dc.clone()
        new_features_dc = features_dc + torch.rand_like(features_dc) * range * 2 - range
        new_features_dc[mask] = features_dc[mask]
        new_features_dc.grad = self._features_dc.grad
        self._features_dc = self.replace_tensor_to_optimizer(new_features_dc, "_features_dc", optimizer, prefix)

    def enlarge_scaling(self, ratio: float = 1.5, threshold: float = 0.02, optimizer: Optimizer = None, prefix: str = ''):
        log(yellow_slim(f'[ENLARGE SCALING] ENLARGE SCALING BY {ratio}'))
        mask = self.get_specular.max(dim=-1).values.flatten() > threshold  # (P,)
        new_scaling = self.scaling_inverse_activation(self.get_scaling * ratio)  # (P, 2)
        new_scaling[mask] = self._scaling[mask]  # (P, 2)
        new_scaling.grad = self._scaling.grad
        self._scaling = self.replace_tensor_to_optimizer(new_scaling, '_scaling', optimizer, prefix)

    def enlarge_opacity(self, enlarge_opacity: float = 0.9, optimizer: Optimizer = None, prefix: str = ''):
        log(yellow_slim(f'[ENLARGE OPACITY] ENLARGE OPACITY TO {enlarge_opacity}'))
        new_opacity = torch.max(self._opacity, self.opacity_inverse_activation(torch.ones_like(self._opacity, ) * enlarge_opacity))
        new_opacity.grad = self._opacity.grad
        self._opacity = self.replace_tensor_to_optimizer(new_opacity, '_opacity', optimizer, prefix)

    def reset_specular(self, reset_specular: float = 0.001, optimizer: Optimizer = None, prefix: str = '', reset_specular_all: bool = False):
        log(yellow_slim(f'[RESET SPECULAR] RESET SPECULAR TO {reset_specular}'))
        if reset_specular_all: new_specular = self.specular_inverse_activation(torch.ones_like(self._specular, ) * reset_specular)
        else: new_specular = torch.min(self._specular, self.specular_inverse_activation(torch.ones_like(self._specular, ) * reset_specular))
        new_specular.grad = self._specular.grad
        self._specular = self.replace_tensor_to_optimizer(new_specular, '_specular', optimizer, prefix)

    def reset_opacity(self, reset_opacity: float = 0.01, optimizer: Optimizer = None, prefix: str = ''):
        log(yellow_slim(f'[RESET OPACITY] RESET OPACITY TO {reset_opacity}'))
        new_opacity = torch.min(self._opacity, self.opacity_inverse_activation(torch.ones_like(self._opacity, ) * reset_opacity))
        self._opacity = self.replace_tensor_to_optimizer(new_opacity, '_opacity', optimizer, prefix)


    def prune_points(self, mask, optimizer: Optimizer, prefix: str = ''):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask, optimizer, prefix)
        for name, new_params in optimizable_tensors.items():
            setattr(self, name.replace(prefix, ''), new_params)
        self.car_gaussian_mask = self.car_gaussian_mask[valid_points_mask]

    def get_xyz_gradient_avg(self):
        avg = self.xyz_gradient_accum / self.denom
        avg[avg.isnan()] = 0.0
        return avg

    def get_xyz_weight_avg(self):
        avg = self.xyz_weight_accum / self.denom
        avg[avg.isnan()] = 0.0
        return avg

    def reset_stats(self):
        device = self.get_xyz.device
        self.xyz_gradient_accum.set_(torch.zeros((self.get_xyz.shape[0], 1), device=device))
        self.denom.set_(torch.zeros((self.get_xyz.shape[0], 1), device=device))
        self.max_radii2D.set_(torch.zeros((self.get_xyz.shape[0]), device=device))
        self.xyz_weight_accum.set_(torch.zeros((self.get_xyz.shape[0], 1), device=device))

    def prune_stats(self, mask):
        valid_points_mask = ~mask
        self.xyz_gradient_accum.set_(self.xyz_gradient_accum[valid_points_mask])
        self.denom.set_(self.denom[valid_points_mask])
        self.max_radii2D.set_(self.max_radii2D[valid_points_mask])
        self.xyz_weight_accum.set_(self.xyz_weight_accum[valid_points_mask])
        assert self.xyz_gradient_accum.shape[0] == self.get_xyz.shape[0]
        assert self.denom.shape[0] == self.get_xyz.shape[0]
        assert self.max_radii2D.shape[0] == self.get_xyz.shape[0]
        assert self.xyz_weight_accum.shape[0] == self.get_xyz.shape[0]



    def build_car_cloned_render_inputs(self, mask, offset=(0.0, 0.0, 0.0)):
        device = self._xyz.device
        dtype = self._xyz.dtype

        offset = torch.as_tensor(offset, device=device, dtype=dtype)
        if offset.ndim != 1 or offset.shape[0] != 3:
            raise ValueError(f"offset must have shape [3], got {offset.shape}")


        clone_xyz = self._xyz[mask] + offset.unsqueeze(0)
        clone_features_dc = self._features_dc[mask]
        clone_features_rest = self._features_rest[mask]
        clone_opacity = self._opacity[mask]
        clone_scaling = self._scaling[mask]
        clone_rotation = self._rotation[mask]
        clone_specular = self._specular[mask]
        clone_roughness = self._roughness[mask]
        clone_neighbor_effect = self._neighbor_effect[mask]
        clone_car_gaussian_mask = self.car_gaussian_mask[mask]

        log(yellow_slim(f'original xyz: {self._xyz.shape}'))
        log(yellow_slim(f'mask xyz: {clone_xyz.shape}'))

        self.render_xyz = torch.cat([self._xyz, clone_xyz], dim=0).detach()
        self.render_features_dc = torch.cat([self._features_dc, clone_features_dc], dim=0).detach()
        self.render_features_rest = torch.cat([self._features_rest, clone_features_rest], dim=0).detach()
        self.render_opacity = torch.cat([self._opacity, clone_opacity], dim=0).detach()
        self.render_scaling = torch.cat([self._scaling, clone_scaling], dim=0).detach()
        self.render_rotation = torch.cat([self._rotation, clone_rotation], dim=0).detach()
        self.render_specular = torch.cat([self._specular, clone_specular], dim=0).detach()
        self.render_roughness = torch.cat([self._roughness, clone_roughness], dim=0).detach()
        self.render_neighbor_effect = torch.cat([self._neighbor_effect, clone_neighbor_effect], dim=0).detach()
        self.render_car_gaussian_mask = torch.cat([self.car_gaussian_mask, clone_car_gaussian_mask], dim=0).detach()

        log(yellow_slim(f'final xyz: {self.render_xyz.shape}'))

    def car_clone_reorder(self, sorted_indices: torch.Tensor, prefix: str = ''):
        """
        Reorder only this object's parameters by filtering optimizer groups with prefix.
        """
        sorted_indices = sorted_indices.long()
        self.render_xyz= self.render_xyz[sorted_indices]
        self.render_features_dc =self.render_features_dc[sorted_indices]
        self.render_features_rest = self.render_features_rest[sorted_indices]
        self.render_opacity= self.render_opacity[sorted_indices]
        self.render_scaling=self.render_scaling[sorted_indices]
        self.render_rotation =self.render_rotation[sorted_indices]
        self.render_specular=self.render_specular[sorted_indices]
        self.render_roughness =self.render_roughness[sorted_indices]
        self.render_neighbor_effect= self.render_neighbor_effect[sorted_indices]
        self.render_car_gaussian_mask = self.render_car_gaussian_mask[sorted_indices]

   

    def prune_visibility(self, optimizer: Optimizer = None, prefix: str = ''):
        n_before = self.get_xyz.shape[0]
        n_after = int(self.max_gs * self.max_gs_threshold)
        n_prune = n_before - n_after

        if n_prune > 0:
            weights = self.get_xyz_weight_avg()
            # Find the mask of top n_prune smallest `self.xyz_weight_accum`
            _, indices = torch.topk(weights[..., 0], n_prune, largest=False)
            prune_mask = torch.zeros((self.get_xyz.shape[0],), dtype=torch.bool, device=self.get_xyz.device)
            prune_mask[indices] = True
            # Prune points
            self.prune_points(prune_mask, optimizer, prefix)
            self.prune_stats(prune_mask)
            torch.cuda.empty_cache()

            log(yellow_slim(f'[PRUNE VISIBILITY] num points pruned: {n_prune}.'))



    def save_ply(
        self, 
        path: str,
        bounds: torch.Tensor = None
    ):
        from plyfile import PlyData, PlyElement
        os.makedirs(dirname(path), exist_ok=True)

        # Only save the points within the bounds
        # `bounds` is a tuple of two 3D points, representing the min and max bounds
        if bounds is not None: mask = ((self._xyz >= bounds[0]) & (self._xyz <= bounds[1])).all(dim=-1)
        else: mask = torch.ones((self._xyz.shape[0],), dtype=torch.bool, device=self._xyz.device)

        xyz = self._xyz[mask].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest[mask].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity[mask].detach().cpu().numpy()
        scales = self._scaling[mask].detach().cpu().numpy()
        rotation = self._rotation[mask].detach().cpu().numpy()
        neighbor_effect= self._neighbor_effect[mask].detach().cpu().numpy()
        car_gaussian_mask=self.car_gaussian_mask[mask].detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, rotation, neighbor_effect,car_gaussian_mask), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

        if self.env_map is not None:
            save_path = path.replace('.ply', '.map')
            torch.save(self.env_map.state_dict(), save_path)

    def load_ply(self, path: str):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        neighbor_effect = np.asarray(plydata.elements[0]["neighbor_effect"])[..., np.newaxis]

        car_gaussian_mask = np.asarray(plydata.elements[0]["car_gaussian_mask"])[..., np.newaxis]

        map_path = path.replace('.ply', '.map')
        if os.path.exists(map_path):
            map_ckpt = torch.load(map_path)
            self.env_map = EnvLightMip(path=None, device='cuda', max_res=map_ckpt['base'].shape[1]).cuda()
            self.env_map.load_state_dict(map_ckpt)
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._neighbor_effect = nn.Parameter(torch.tensor(neighbor_effect, dtype=torch.float, device="cuda").requires_grad_(True))
        self.car_gaussian_mask = nn.Parameter(torch.tensor(car_gaussian_mask, dtype=torch.float, device="cuda").requires_grad_(False))
        self.active_sh_degree = make_buffer(torch.full((1,), self.max_sh_degree, dtype=torch.long))



def car_render(
    viewpoint_camera,
    pc: GaussianModel,
    hash,
    pipe: dotdict,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color: torch.Tensor = None,
    device: str = 'cuda'
):
    # Lazy import to avoid circular import
    if pc.render_reflection and pc.specular_channels == 1: 

        from diff_surfel_rasterization_wet_ch05 import GaussianRasterizationSettings, GaussianRasterizer
    elif pc.render_reflection and pc.specular_channels == 3: 
        from diff_surfel_rasterization_wet_ch07 import GaussianRasterizationSettings, GaussianRasterizer

    else: 
        from diff_surfel_rasterization_wet import GaussianRasterizationSettings, GaussianRasterizer


    # Create zero tensor, we will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.render_xyz, requires_grad=True, device=device) + 0
    try: screenspace_points.retain_grad()
    except: pass

    # Set up rasterization configuration

    car_image_masks=viewpoint_camera.car_mask
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    # Get the means, opacities, and colors of the Gaussians
    means3D = pc.render_xyz
    neighbor_effects=pc._neighbor_effect
    means2D = screenspace_points
    opacity = pc.get_render_opacity
    car_gaussian_masks=pc.render_car_gaussian_mask
 
    starts = hash.flat_starts.to(device)
    ends = hash.flat_ends.to(device)
    densities = None
    
    log(yellow_slim(f'before render xyz: {means3D.shape}'))
    # If precomputed 3d covariance is provided, use it
    # If not, then it will be computed from scaling / rotation by the rasterizer
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # NOTE: Currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0, 1, 3]] @ world2pix[:, [0, 1, 3]]).permute(0, 2, 1).reshape(-1, 9)  # `glm` is column major
    else:
        scales = pc.get_render_scaling
        rotations = pc.get_render_rotation
        normals = quat_to_normal_z(rotations)
        gaussian_envlight = None
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_render_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.render_xyz - viewpoint_camera.camera_center.repeat(pc.get_render_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_render_features
    else:
        colors_precomp = override_color

    # Additional reflection-related splatting variable
    if pc.render_reflection and colors_precomp is not None:
        colors_precomp = torch.cat([colors_precomp, pc.get_render_specular, pc.get_render_specular], dim=-1)  # (P, C+2)
    elif pc.render_reflection:
        raise ValueError('Reflection is enabled but no color is provided.')

    # Create the rasterizer and perform the rendering
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)


    rendered_image, radii, allmap, weight = rasterizer(
        gaussian_envlight,
        neighbor_effects=neighbor_effects,
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        starts=starts,
        ends=ends,
        densities=densities,
        car_image_masks=car_image_masks,
        car_gaussian_masks =  car_gaussian_masks
    )

    # Prepare the output dictionary
    output = dotdict(render=rendered_image[:3])  # (3, H, W)
    if pc.render_reflection:
        output.update(dotdict(
            specular=rendered_image[3:3+pc.specular_channels],  # (1, H, W)
            roughness=rendered_image[3+pc.specular_channels:3+pc.specular_channels+1]  # (1, H, W)
        ))
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    output.update(dotdict(#
        viewspace_points=means2D,  # (P, 3)
        visibility_filter=radii>0,  # (P,)
        radii=radii,  # (P,)
        weight_accumulate=weight.detach().clone()  # (P,)
    ))

    render_neighbor_effects=allmap[7:8]
    render_neighbor_percent=allmap[8:9]
    render_neighbor_indirect=allmap[9:10]
    # Post-process additional rendered maps for regularizations
    # Get the rendered alpha map
    render_alpha = allmap[1:2]  # (1, H, W)
    # Get the rendered normal map
    render_normal = allmap[2:5]  # (3, H, W)
    # Transform normal from view space to world space
    render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_view_transform[:3, :3].T)).permute(2, 0, 1)  # (3, H, W)

    # Get the rendered median depth map
    render_depth_median = allmap[5:6]  # (1, H, W)
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)  # (1, H, W)
    # Get the rendered expected depth map
    render_depth_expect = allmap[0:1]  # (1, H, W)
    render_depth_expect = (render_depth_expect / render_alpha)  # (1, H, W)
    render_depth_expect = torch.nan_to_num(render_depth_expect, 0, 0)  # (1, H, W)

    # Psedo surface attributes, surface depth is either median or expected by setting depth_ratio to 1 or 0
    # - for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # - for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surface_depth = render_depth_expect * (1 - pipe.depth_ratio) + render_depth_median * pipe.depth_ratio  # (1, H, W)
    # Assume the depth points form the 'surface' and generate psudo surface normal for regularizations
    surface_normal = dpt2norm(viewpoint_camera, surface_depth)  # (H, W, 3)
    surface_normal = surface_normal.permute(2, 0, 1)  # (3, H, W)
    # Remember to multiply with accum_alpha since render_normal is unnormalized
    surface_normal = surface_normal * (render_alpha).detach()  # (3, H, W)

    # Get the rendered depth distortion map
    render_distortion = allmap[6:7]  # (1, H, W)

    # Update the output dictionary
    output.update(dotdict(
        rend_alpha=render_alpha,  # (1, H, W)
        rend_normal=render_normal,  # (3, H, W)
        rend_dist=render_distortion,  # (1, H, W)
        render_neighbor_effects=render_neighbor_effects,#new
        render_neighbor_percent=render_neighbor_percent,#new
        render_neighbor_indirect=render_neighbor_indirect,#new
        surf_depth=surface_depth,  # (1, H, W)
        surf_normal=surface_normal  # (3, H, W)
    ))

    return output
 

def dpt2xyz(
    camera,
    dpt: torch.Tensor,
    device: str = 'cuda'  
): 
    # Get the camera extrinsic matrix
    c2w = (camera.world_view_transform.T).inverse()
    # Get the camera intrinsic matrix
    W, H = camera.image_width, camera.image_height
    fx = W / (2 * math.tan(camera.FoVx / 2.))
    fy = H / (2 * math.tan(camera.FoVy / 2.))
    K = torch.tensor([
        [fx, 0., W/2.],
        [0., fy, H/2.],
        [0., 0., 1.0]
    ]).float().to(device, non_blocking=True)  # (3, 3)

    # Backproject the depth map to 3D points
    u, v = torch.meshgrid(
        torch.arange(W).float().to(device, non_blocking=True),
        torch.arange(H).float().to(device, non_blocking=True),
        indexing='xy'
    )  # (H, W), (H, W)
    xyz = torch.stack(
        [u, v, torch.ones_like(u)], dim=-1
    ).reshape(-1, 3)  # (H * W, 3)
    ray_d = xyz @ K.inverse().mT @ c2w[:3, :3].mT  # (H * W, 3)
    ray_o = c2w[:3, 3]  # (3,)
    xyz = dpt.reshape(-1, 1) * ray_d + ray_o  # (H * W, 3)
    return xyz


def dpt2norm(
    camera,
    dpt: torch.Tensor,
    device: str = 'cuda'
):
    # Convert the depth map to 3D points
    xyz = dpt2xyz(
        camera, dpt, device
    ).reshape(*dpt.shape[1:], 3)  # (H, W, 3)

    out = torch.zeros_like(xyz)  # (H, W, 3)
    # Compute the normal map from the depth map
    dx = torch.cat([xyz[2:, 1:-1] - xyz[:-2, 1:-1]], dim=0)  # (H-2, W-2, 3)
    dy = torch.cat([xyz[1:-1, 2:] - xyz[1:-1, :-2]], dim=1)  # (H-2, W-2, 3)
    norm = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1)  # (H-2, W-2, 3)
    out[1:-1, 1:-1, :] = norm  # (H, W, 3)
    return out