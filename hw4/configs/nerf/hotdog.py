from copy import deepcopy
_base_ = '../default.py'

expname = ''
basedir = ''

data = dict(
    # datadir='../hw4_data/hotdog',
    datadir = '',
    dataset_type='blender',
    white_bkgd=True,
)

coarse_train = dict(
    N_iters=10000,                 # number of optimization steps
    N_rand=10000,                  # batch size (number of random rays per optimization step)
    lrate_density=0.12,           # lr of density voxel grid
    lrate_k0=0.12,                # lr of color/feature voxel grid
    lrate_rgbnet=1e-3,            # lr of the mlp to preduct view-dependent color
    lrate_decay=20,               # lr decay by 0.1 after every lrate_decay*1000 steps
    pervoxel_lr=True,             # view-count-based lr
    pervoxel_lr_downrate=1,       # downsampled image for computing view-count-based lr
    ray_sampler='random',         # ray sampling strategies
    weight_main=1.0,              # weight of photometric loss
    weight_entropy_last=0.01,     # weight of background entropy loss
    weight_nearclip=0,
    weight_distortion=0,
    weight_rgbper=0.1,            # weight of per-point rgb loss
    tv_every=1,                   # count total variation loss every tv_every step
    tv_after=0,                   # count total variation loss from tv_from step
    tv_before=0,                  # count total variation before the given number of iterations
    tv_dense_before=0,            # count total variation densely before the given number of iterations
    weight_tv_density=0.0,        # weight of total variation loss of density voxel grid
    weight_tv_k0=0.0,             # weight of total variation loss of color/feature voxel grid
    pg_scale=[],                  # checkpoints for progressive scaling
    decay_after_scale=1.0,        # decay act_shift after scaling
    skip_zero_grad_fields=[],     # the variable name to skip optimizing parameters w/ zero grad in each iteration
    maskout_lt_nviews=0,
)

fine_train = deepcopy(coarse_train)
fine_train.update(dict(
    N_iters=25000,
    pervoxel_lr=False,
    ray_sampler='in_maskcache',
    weight_entropy_last=0.001,
    weight_rgbper=0.01,
    pg_scale=[1000, 2000, 3000, 4000],
    skip_zero_grad_fields=['density', 'k0'],
))

''' Template of model and rendering options
'''
coarse_model_and_render = dict(
    num_voxels=1200000,           # expected number of voxel
    num_voxels_base=1200000,      # to rescale delta distance
    density_type='DenseGrid',     # DenseGrid, TensoRFGrid
    k0_type='DenseGrid',          # DenseGrid, TensoRFGrid
    density_config=dict(),
    k0_config=dict(),
    mpi_depth=128,                # the number of planes in Multiplane Image (work when ndc=True)
    nearest=False,                # nearest interpolation
    pre_act_density=False,        # pre-activated trilinear interpolation
    in_act_density=False,         # in-activated trilinear interpolation
    bbox_thres=1e-3,              # threshold to determine known free-space in the fine stage
    mask_cache_thres=1e-3,        # threshold to determine a tighten BBox in the fine stage
    rgbnet_dim=0,                 # feature voxel grid dim
    rgbnet_full_implicit=False,   # let the colors MLP ignore feature voxel grid
    rgbnet_direct=True,           # set to False to treat the first 3 dim of feature voxel grid as diffuse rgb
    rgbnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=128,             # width of the colors MLP
    alpha_init=1e-6,              # set the alpha values everywhere at the begin of training
    fast_color_thres=1e-7,        # threshold of alpha value to skip the fine stage sampled point
    maskout_near_cam_vox=True,    # maskout grid points that between cameras and their near planes
    world_bound_scale=1,          # rescale the BBox enclosing the scene
    stepsize=0.5,                 # sampling stepsize in volume rendering
)

fine_model_and_render = deepcopy(coarse_model_and_render)
fine_model_and_render.update(dict(
    num_voxels=180**3,
    num_voxels_base=180**3,
    rgbnet_dim=12,
    alpha_init=1e-2,
    fast_color_thres=1e-4,
    maskout_near_cam_vox=False,
    world_bound_scale=1.05,
))

del deepcopy