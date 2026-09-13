"""Independent PPO run with the arm-hand MLP and optional restoration features."""

from isaaclab.utils import configclass
from .rsl_rl_ppo_cfg_rotate_bulb import DexsuiteRevo3RotateBulbPPORunnerCfg


@configclass
class DexsuiteRevo3InsertDPegPPORunnerCfg(DexsuiteRevo3RotateBulbPPORunnerCfg):
    experiment_name = "dexsuite_revo3_insert_d_peg_v3"
    save_interval = 25
    policy = DexsuiteRevo3RotateBulbPPORunnerCfg().policy.copy()
    policy.init_noise_std = 0.15
    algorithm = DexsuiteRevo3RotateBulbPPORunnerCfg().algorithm.copy()
    algorithm.learning_rate = 3.0e-5
    # Sparse, high-variance D-Peg rollouts make adaptive KL scheduling
    # unstable: one outlier minibatch can multiply the rate and explode the
    # residual policy. Keep updates at the calibrated rate.
    algorithm.schedule = "fixed"
    algorithm.entropy_coef = 0.001
    # At 30 Hz, a reward delayed by the full 15-second episode retains about 64% weight.
    algorithm.gamma = 0.999
