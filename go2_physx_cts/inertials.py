"""Match diagnostic MuJoCo inertials to the actual training USD (fixed links included)."""
import json
from pathlib import Path
import numpy as np
import mujoco

def apply_usd_inertials(model):
    record=json.loads((Path(__file__).parent/'assets/go2_usd_inertials.json').read_text())
    for name,row in record['bodies'].items():
        i=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        if i<0:raise ValueError(f'Missing diagnostic body {name}')
        model.body_mass[i]=row['mass'];model.body_ipos[i]=row['com']
        model.body_inertia[i]=row['diagonal_inertia'];model.body_iquat[i]=row['principal_axes_wxyz']
    if abs(model.body_mass.sum()-record['nominal_total_mass_kg'])>1e-5:raise ValueError('Diagnostic mass mismatch')
    return record
