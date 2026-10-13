"""Observation-anchored floating polarization bridge (research candidate).

Keep official quantized grayscale for the raw observation, then add the
enhancer's floating luminance correction. Zero correction exactly preserves
the official input. No claim of downstream benefit follows from this contract.
"""
import torch

def gray_correction(polar, observation):
    if polar.shape != observation.shape or polar.ndim != 4 or polar.shape[1] != 12:
        raise ValueError('Matching [B,12,H,W] polarization tensors required')
    weights = polar.new_tensor([.299,.587,.114]).view(1,3,1,1)
    delta = polar.clamp(0,1) - observation.clamp(0,1)
    return [(x * weights).sum(1,keepdim=True) for x in delta.chunk(4,1)]

def anchored_task_inputs(polar, observation, image):
    from rgb_final_confirmation.polarfree import opencv_rgb_to_gray_ste
    if image.shape != (polar.shape[0],3,*polar.shape[-2:]):
        raise ValueError('RGB shape must match polarization')
    raw_gray = [opencv_rgb_to_gray_ste(x) for x in observation.chunk(4,1)]
    gray = [(g+d).clamp(0,1) for g,d in zip(raw_gray,gray_correction(polar,observation))]
    g0,g45,g90,g135 = [x.double() for x in gray]
    # Preserve the released PolarFree Stokes convention and numerical guards.
    intensity = .5*(g0+g45+g90+g135)+1e-4
    q,u = g0-g90,g45-g135
    q = torch.where(q==0,q.new_tensor(1e-4),q)
    intensity = torch.where(intensity==0,intensity.new_tensor(1e-4),intensity)
    dolp=(torch.sqrt(q*q+u*u)/intensity).clamp(max=1).to(polar.dtype)
    aolp=(.5*torch.atan2(u,q)).to(polar.dtype)
    return [image,*gray,aolp,dolp]
