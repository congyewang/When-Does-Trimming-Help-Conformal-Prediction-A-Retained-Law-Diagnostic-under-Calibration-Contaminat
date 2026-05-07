from __future__ import annotations
import numpy as np
def derive_seed(base_seed: int, replicate: int, stage: int) -> int:
    sequence = np.random.SeedSequence([base_seed, replicate, stage])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])
def make_rng(base_seed: int, replicate: int, stage: int) -> np.random.Generator:
    return np.random.default_rng(derive_seed(base_seed, replicate, stage))
