"""GPU-resident port of the supplied RewardThresholdCurriculum.

Each world owns its weights. Reversed sampling bounds are corrected, samples
are bounded by command limits, and completed command windows reset their sums.
"""
import torch


class RewardThresholdCurriculum:
    def __init__(self, num_envs, limits, bins, initial, device, generator):
        self.device, self.generator = device, generator
        axes=[torch.linspace(-limit,limit,count,device=device) for limit,count in zip(limits,bins)]
        self.grid=torch.stack(torch.meshgrid(*axes,indexing='ij'),-1).reshape(-1,3)
        self.limits=self.grid.new_tensor(limits)
        self.half_bin=self.grid.new_tensor([2*limit/(count-1)/2 for limit,count in zip(limits,bins)])
        active=(self.grid.abs()<=self.grid.new_tensor(initial).clamp_max(self.limits)).all(-1)
        # Very small initial support can exclude every coarse grid centroid.
        if not bool(active.any()):
            active[self.grid.square().sum(-1).argmin()]=True
        self.weights=active.float().expand(num_envs,-1).clone()

    def update(self, ids, bins, linear, angular, linear_threshold=.8,
               angular_threshold=.5, local_range=.5, increment=.2):
        if ids.numel()==0:
            return
        success=(linear>linear_threshold)&(angular>angular_threshold)
        old=self.weights[ids]
        rows=torch.arange(ids.numel(),device=self.device)
        old[rows,bins]=(old[rows,bins]+increment*success).clamp_max(1)
        adjacent=(self.grid[None]-self.grid[bins,None]).abs().le(local_range).all(-1)
        old.add_(increment*(adjacent&success[:,None])).clamp_max_(1)
        self.weights[ids]=old

    def sample(self, ids):
        chosen=torch.multinomial(self.weights[ids],1,generator=self.generator).squeeze(-1)
        centers=self.grid[chosen]
        low=torch.maximum(centers-self.half_bin,-self.limits)
        high=torch.minimum(centers+self.half_bin,self.limits)
        samples=low+(high-low)*torch.rand(centers.shape,device=self.device,generator=self.generator)
        return samples,chosen
