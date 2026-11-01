"""Read existing native contact solutions; never modify the physical model."""
import re

import mujoco
import numpy as np

FINGERS=('thumb','index','middle','ring','little')
PAIRS=tuple((i,j) for i in range(5) for j in range(i+1,5))
FEATURE_NAMES=tuple('ball_'+n for n in FINGERS)+tuple(FINGERS[i]+'_'+FINGERS[j] for i,j in PAIRS)


class ContactLoadReader:
    def __init__(self,model):
        self.model=model
        self.ball=model.geom('ball/surface').id
        self.finger=np.full(model.ngeom,-1,dtype=np.int64)
        for g in range(model.ngeom):
            match=re.match(r'hand/right_(thumb|index|middle|ring|little)_',model.geom(g).name or '')
            if match:self.finger[g]=FINGERS.index(match[1])
        assert set(self.finger)==set(range(-1,5))
        self.pair_index={pair:5+k for k,pair in enumerate(PAIRS)}

    def forces(self,data):
        result=np.zeros(15,dtype=np.float64)
        force=np.zeros(6,dtype=np.float64)
        for k,contact in enumerate(data.contact):
            a,b=map(int,contact.geom);fa,fb=self.finger[a],self.finger[b]
            if a==self.ball and fb>=0:index=int(fb)
            elif b==self.ball and fa>=0:index=int(fa)
            elif fa>=0 and fb>=0 and fa!=fb:index=self.pair_index[tuple(sorted((int(fa),int(fb))))]
            else:continue
            mujoco.mj_contactForce(self.model,data,k,force)
            result[index]+=max(0.,float(force[0]))
        assert np.isfinite(result).all()
        return result

    def features(self,data):
        # Fixed physical units: sum of normal magnitudes, never net force.
        return np.clip(self.forces(data)/20.,0.,4.).astype(np.float32)
