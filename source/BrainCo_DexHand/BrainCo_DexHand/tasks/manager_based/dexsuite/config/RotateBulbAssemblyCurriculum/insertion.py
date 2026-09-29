"""Original insertion shaping, eligible only after physical reset validation."""
from ..RotateBulbCustom.assembly_guidance import AssemblyGuidanceReward


class CurriculumInsertionReward(AssemblyGuidanceReward):
    def __call__(self, env, component: str):
        reward = super().__call__(env, component)
        runtime = getattr(env, "_bulb_curriculum", None)
        if runtime is None:
            return reward * 0.
        valid = runtime.validated & ~runtime.reset_failed
        self.hover_reached &= valid
        return reward * valid
