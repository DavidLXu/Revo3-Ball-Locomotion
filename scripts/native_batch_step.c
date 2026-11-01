/* Batch the existing native algorithm, retaining independent mjData state. */
#include <mujoco/mujoco.h>
#include <math.h>
#include <string.h>

int batch_step_version(void) { return mj_version(); }

int batch_step(const mjModel* m, mjData** ds, int n, int count, int ball_dof,
               int threads, int ball_geom, const int* fingers,
               const unsigned char* reset, double* qpos, double* qvel,
               const double* ctrl, const double* xfrc, double* xpos,
               double* xmat, double* sensors, double* costs, float* loads) {
  int bad = 0;
  #pragma omp parallel for num_threads(threads) schedule(static) reduction(|:bad)
  for (int i = 0; i < n; ++i) {
    mjData* d = ds[i];
    if (reset[i]) {
      memcpy(d->qpos, qpos + i*m->nq, m->nq*sizeof(mjtNum));
      memcpy(d->qvel, qvel + i*m->nv, m->nv*sizeof(mjtNum));
    }
    memcpy(d->ctrl, ctrl + i*m->nu, m->nu*sizeof(mjtNum));
    memcpy(d->xfrc_applied, xfrc + i*m->nbody*6, m->nbody*6*sizeof(mjtNum));
    double total = 0.;
    for (int j = 0; j < count; ++j) {
      mj_step(m, d);
      if (ball_dof >= 0) {
        const double* v = d->qvel;
        double angular = v[3]*v[3] + v[4]*v[4] + v[5]*v[5];
        double relative = (v[2]-v[ball_dof+2])/.08;
        double vertical = relative*relative;
        total += (angular > 16. ? 16. : angular) + (vertical > 16. ? 16. : vertical);
      }
    }
    if (count && ball_dof >= 0) costs[i] = total/count;
    mj_forward(m, d);
    if (loads) {
      double sums[15] = {0};
      for (int k = 0; k < d->ncon; ++k) {
        int a = d->contact[k].geom[0], b = d->contact[k].geom[1];
        int fa = a >= 0 ? fingers[a] : -1, fb = b >= 0 ? fingers[b] : -1;
        int index = -1;
        if (a == ball_geom && fb >= 0) index = fb;
        else if (b == ball_geom && fa >= 0) index = fa;
        else if (fa >= 0 && fb >= 0 && fa != fb) {
          int low = fa < fb ? fa : fb, high = fa < fb ? fb : fa;
          index = 5 + low*(9-low)/2 + high-low-1;
        }
        if (index >= 0) {
          mjtNum force[6];
          mj_contactForce(m, d, k, force);
          sums[index] += force[0] > 0. ? force[0] : 0.;
        }
      }
      for (int j = 0; j < 15; ++j) {
        if (!isfinite(sums[j])) bad = 1;
        double value = sums[j]/20.;
        loads[i*15+j] = (float)(value > 4. ? 4. : value);
      }
    }
    memcpy(qpos+i*m->nq, d->qpos, m->nq*sizeof(mjtNum));
    memcpy(qvel+i*m->nv, d->qvel, m->nv*sizeof(mjtNum));
    memcpy(xpos+i*m->nbody*3, d->xpos, m->nbody*3*sizeof(mjtNum));
    memcpy(xmat+i*m->nbody*9, d->xmat, m->nbody*9*sizeof(mjtNum));
    memcpy(sensors+i*m->nsensordata, d->sensordata, m->nsensordata*sizeof(mjtNum));
  }
  return bad;
}
