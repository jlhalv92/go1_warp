"""Per-world geom-group collision queries for MuJoCo Warp 3.16.

No physics state, reward, action mapping or termination behavior is modified.
A snapshot can serve many queries with one contact-count readback.
"""
import math
import numpy as np
import torch
import mujoco


# Match the groups in the supplied original environment (calves use calf1 only).
DEFAULT_COLLISION_GROUPS = {
    'floor': ['floor'],
    'feet': ['FR_foot', 'FL_foot', 'RR_foot', 'RL_foot'],
    **{f'{leg}_foot': [f'{leg}_foot'] for leg in ('FR', 'FL', 'RR', 'RL')},
    **{f'{leg}_calf': [f'{leg}_calf1'] for leg in ('FR', 'FL', 'RR', 'RL')},
    **{f'{leg}_thigh': [f'{leg}_thigh{i}' for i in (1, 2, 3)]
       for leg in ('FR', 'FL', 'RR', 'RL')},
    'trunk': [f'trunk_{i}' for i in range(1, 9)],
    **{f'trunk_{i}': [f'trunk_{i}'] for i in (1, 2, 3)},
}


class ContactSnapshot:
    """Contact identities captured at one collision evaluation; query before/after reset safely.

    Results are bool tensors of shape (num_envs,), on the detector's device.
    Contacts with invalid world/geom IDs are excluded, not indexed as negative IDs.
    """
    def __init__(self, detector, geom, worldid, dist, max_distance):
        self.detector = detector
        self.geom = geom.to(device=detector.device, dtype=torch.long, copy=True)
        self.worldid = worldid.to(device=detector.device, dtype=torch.long, copy=True)
        self.dist = dist.to(device=detector.device, dtype=torch.float32, copy=True)
        g1, g2 = self.geom.unbind(-1)
        self.valid = ((g1 >= 0) & (g1 < detector.model.ngeom)
                      & (g2 >= 0) & (g2 < detector.model.ngeom)
                      & (self.worldid >= 0) & (self.worldid < detector.num_envs))
        if max_distance is not None:
            if math.isnan(max_distance):
                raise ValueError('max_distance cannot be NaN')
            self.valid &= self.dist <= max_distance
        # Clamp only indices to allow vectorized lookup on invalid contacts.
        # The validity mask still prevents those contacts from contributing.
        self.g1 = g1.clamp(0, detector.model.ngeom - 1)
        self.g2 = g2.clamp(0, detector.model.ngeom - 1)
        self.world = self.worldid.clamp(0, detector.num_envs - 1)

    def _reduce(self, match):
        counts = torch.zeros(self.detector.num_envs, dtype=torch.int32, device=self.detector.device)
        # Accumulation handles any number of matching contacts per world correctly.
        counts.scatter_add_(0, self.world, (match & self.valid).to(torch.int32))
        return counts > 0

    def check_collision(self, groups1, groups2):
        """Match either geom order. Each argument is a group name or list of names."""
        mask1 = self.detector.group_mask(groups1)
        mask2 = self.detector.group_mask(groups2)
        return self._reduce((mask1[self.g1] & mask2[self.g2])
                            | (mask1[self.g2] & mask2[self.g1]))

    def check_any_collision(self, groups):
        """Contact between a named group (or union) and any other geom."""
        mask = self.detector.group_mask(groups)
        return self._reduce(mask[self.g1] | mask[self.g2])

    def check_pairs(self, pairs):
        """Return bool (num_envs, len(pairs)), preserving the input pair order."""
        results = [self.check_collision(a, b) for a, b in pairs]
        if not results:
            return torch.empty((self.detector.num_envs, 0), dtype=torch.bool,
                               device=self.detector.device)
        return torch.stack(results, dim=-1)


class CollisionDetector:
    def __init__(self, model, num_envs, device, groups=None):
        self.model = model
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        if self.num_envs < 1 or model.ngeom < 1:
            raise ValueError('num_envs and model.ngeom must be positive')
        self.groups = {}
        self._masks = {}
        self._union_masks = {}
        definitions = dict(DEFAULT_COLLISION_GROUPS if groups is None else groups)
        for name, geom_names in definitions.items():
            if isinstance(geom_names, str):
                geom_names = [geom_names]
            ids = set()
            for geom_name in geom_names:
                gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
                if gid < 0:
                    raise ValueError(f'Collision group {name!r}: geom {geom_name!r} not found in model')
                ids.add(gid)
            self.groups[name] = frozenset(ids)
            mask = torch.zeros(model.ngeom, dtype=torch.bool, device=self.device)
            if ids:
                mask[list(ids)] = True
            self._masks[name] = mask

    def group_mask(self, groups):
        names = (groups,) if isinstance(groups, str) else tuple(sorted(set(groups)))
        if names not in self._union_masks:
            mask = torch.zeros(self.model.ngeom, dtype=torch.bool, device=self.device)
            for name in names:
                if name not in self._masks:
                    raise KeyError(f'Unknown collision group {name!r}; available: {list(self.groups)}')
                mask |= self._masks[name]
            self._union_masks[names] = mask
        return self._union_masks[names]

    @torch.no_grad()
    def snapshot(self, data, *, max_distance=None):
        """Read contact identities; None matches all generated contacts, including margins.

        data: native mujoco_warp.Data, or a list of CPU MjData, one per world.
        max_distance=0.0 restricts detection to touching/penetrating contacts.
        Warp contact count nacon is read once (a GPU/CPU synchronization point).
        Snapshots own their identity/distance tensors and survive later steps/resets.
        """
        if isinstance(data, (list, tuple)):
            if len(data) != self.num_envs:
                raise ValueError('CPU MjData list must contain one item per world')
            geoms, worlds, distances = [], [], []
            for world, d in enumerate(data):
                for i in range(d.ncon):
                    c = d.contact[i]
                    geoms.append((int(c.geom1), int(c.geom2)))
                    worlds.append(world)
                    distances.append(float(c.dist))
            geom = torch.as_tensor(np.asarray(geoms, dtype=np.int64).reshape(-1, 2))
            worldid = torch.as_tensor(np.asarray(worlds, dtype=np.int64))
            dist = torch.as_tensor(np.asarray(distances, dtype=np.float32))
        else:
            import warp as wp
            if not hasattr(data, 'nacon'):
                raise TypeError('Expected native MuJoCo Warp Data with an nacon contact counter')
            if data.nworld != self.num_envs:
                raise ValueError('Warp world count does not match detector')
            count = int(wp.to_torch(data.nacon).reshape(-1)[0].item())
            capacity = data.contact.geom.shape[0]
            if count < 0 or count > capacity:
                raise RuntimeError(f'Contact buffer overflow: {count} contacts, capacity {capacity}; '
                                   'increase nconmax/naconmax before querying collisions')
            geom = wp.to_torch(data.contact.geom)[:count]
            worldid = wp.to_torch(data.contact.worldid)[:count]
            dist = wp.to_torch(data.contact.dist)[:count]
        return ContactSnapshot(self, geom, worldid, dist, max_distance)

    def check_collision(self, data, groups1, groups2, *, max_distance=None):
        return self.snapshot(data, max_distance=max_distance).check_collision(groups1, groups2)

    def check_any_collision(self, data, groups, *, max_distance=None):
        return self.snapshot(data, max_distance=max_distance).check_any_collision(groups)
