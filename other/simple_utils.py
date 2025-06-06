import numpy as np
import pandas as pd
import random

# %%

def set_random_seed(random_seed) -> None:
    np.random.seed(random_seed)
    random.seed(random_seed)