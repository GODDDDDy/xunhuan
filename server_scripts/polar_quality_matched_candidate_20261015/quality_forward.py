"""All-path quality-condition diagnostic of the archived Experiment 22 model.

The frozen, FP32, two-round generation path is copied from study.recurrent_forward.
Only explicit b/C inputs to candidate generation may change. Neither the legacy
acceptance head nor the final action-cost head is executed. GT is never read.

Load the archived source modules/model in the caller, then:
    first = shared_first(model, task, batch, study_module=study)
    out = candidate_from_first(model, task, first, 'real_quality',
                               study_module=study)
    # out.ungated_t2 is U; out.t1 is the unchanged first-round output.

Do not run candidates concurrently on one model: temporary observation hooks are
installed on that model. SharedFirst's tensors/dictionaries are never cleared.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import math
from typing import Any, Dict, Optional

import torch
from torch.nn import functional as F


CONDITIONS = ('real_quality', 'fixed_quality', 'shifted_quality')


def _study(module):
    return module if module is not None else importlib.import_module('study')


def _ensure_evaluation(model, task):
    if model.training or task.training:
        raise ValueError('This helper is a frozen-model evaluation diagnostic.')
    if not math.isfinite(float(model.tau)) or float(model.tau) <= 0:
        raise ValueError('The inherited confidence temperature must be positive.')


def tensor_summary(value):
    """JSON-safe summaries; no image encoding, clipping, or tensor mutation."""
    x = value.detach().float()
    values = torch.stack((x.amin(), x.amax(), x.mean(),
                          x.std(unbiased=False))).cpu().tolist()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                minimum=values[0], maximum=values[1], mean=values[2],
                std=values[3], finite=bool(torch.isfinite(x).all()))


def _rng_snapshot(is_cuda):
    cpu = torch.get_rng_state().clone()
    cuda = [x.clone() for x in torch.cuda.get_rng_state_all()] if is_cuda else None
    return cpu, cuda


@contextmanager
def _fork_from(cpu, cuda):
    devices = list(range(len(cuda))) if cuda is not None else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
        yield


@dataclass
class SharedFirst:
    polar: torch.Tensor
    image: torch.Tensor
    enhanced: Any
    t1: torch.Tensor
    features: Dict[str, torch.Tensor]
    polar_summary: torch.Tensor
    error_scale: torch.Tensor
    confidence_map: torch.Tensor
    task_rng_cpu: torch.Tensor
    task_rng_cuda: Optional[list]
    candidate_rng_cpu: torch.Tensor
    candidate_rng_cuda: Optional[list]
    tau: float
    need_alignment: bool


@torch.no_grad()
def shared_first(model, task, batch, need_alignment=False, *, study_module=None):
    """Execute the original E1/task1/H exactly once; store both RNG boundaries.

    Only batch['polar'] and batch['official_rgb'] are accessed. task_rng is captured
    after E1 and before task1, exactly as in the original second-task fork. The
    post-H state starts each candidate branch and is restored after that branch.
    """
    _ensure_evaluation(model, task)
    study = _study(study_module)
    p0, image = batch['polar'], batch['official_rgb']
    enhanced = study.enhance(model, p0, p0, image, project=need_alignment)
    task_cpu, task_cuda = _rng_snapshot(p0.is_cuda)
    t1, features = study.task_forward(model, task, enhanced.refined_polar, image, p0)
    summary = study.polarization_summary(p0)
    b = model.confidence_head(image, t1, summary, features)
    c = torch.exp(-b / model.tau).detach()
    if b.ndim != 4 or b.shape[1] != 1 or not torch.isfinite(b).all() or (b <= 0).any():
        raise ValueError('H must return a finite, positive Bx1xhxw error map.')
    candidate_cpu, candidate_cuda = _rng_snapshot(p0.is_cuda)
    return SharedFirst(p0, image, enhanced, t1, dict(features), summary, b, c,
                       task_cpu, task_cuda, candidate_cpu, candidate_cuda,
                       float(model.tau), bool(need_alignment))


def quality_conditions(first, condition, fixed_b0=None):
    """Build matched b/C controls without consuming any random-number stream.

    Fixed b0 must be caller-calibrated from training predictions, not validation
    outcomes. Shift uses the H-native map grid and one predeclared offset: half of
    each dimension, rounded down. Both maps are rolled together.
    """
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown quality condition: {condition!r}')
    if condition == 'real_quality':
        b, c = first.error_scale, first.confidence_map
        details = dict(kind=condition)
    elif condition == 'fixed_quality':
        if fixed_b0 is None:
            raise ValueError('fixed_quality requires a training-calibrated b0.')
        value = float(fixed_b0)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Training-calibrated b0 must be finite and positive.')
        b = torch.full_like(first.error_scale, value)
        c = torch.exp(-b / first.tau).detach()
        details = dict(kind=condition, b0=value, definition='C0=exp(-b0/tau)')
    else:
        h, w = first.error_scale.shape[-2:]
        offset = (h // 2, w // 2)
        b = torch.roll(first.error_scale, shifts=offset, dims=(-2, -1))
        c = torch.roll(first.confidence_map, shifts=offset, dims=(-2, -1))
        details = dict(kind=condition, native_grid=[h, w], offset=list(offset),
                       transformation='paired torch.roll; no RNG consumption')
    return b, c, details


class _PathAudit:
    """Read-only hooks record what the real consuming modules actually receive."""
    def __init__(self, model, first, b, c, gate, enabled):
        self.model, self.first = model, first
        self.b, self.c, self.gate, self.enabled = b, c, gate, enabled
        self.events, self.handles, self.methods = [], [], []

    @torch.no_grad()
    def record(self, name, actual, expected):
        if not self.enabled:
            return
        if actual.shape != expected.shape:
            raise AssertionError(f'{name}: controlled condition shape differs')
        equal = torch.equal(actual, expected)
        self.events.append(dict(path=name, received=tensor_summary(actual),
                                expected=tensor_summary(expected), equal=equal,
                                maximum_absolute_difference=float((actual-expected).abs().max())))
        if not equal:
            raise AssertionError(f'{name}: a candidate path used a different quality condition')

    @staticmethod
    def resized(value, size):
        return F.interpolate(value, size=size, mode='bilinear', align_corners=False)

    def patch_method(self, owner, name, wrapper):
        own_attribute = owner.__dict__.get(name, None)
        had_own_attribute = name in owner.__dict__
        original = getattr(owner, name)
        setattr(owner, name, wrapper(original))
        self.methods.append((owner, name, had_own_attribute, own_attribute))

    def install(self):
        if not self.enabled:
            return
        feedback = self.model.feedback

        def make_wrapper(original):
            def observed(*args, **kwargs):
                actual = args[5] if len(args) > 5 else kwargs['confidence']
                self.record('feedback.make_context.confidence', actual, self.gate)
                return original(*args, **kwargs)
            return observed
        self.patch_method(feedback, 'make_context', make_wrapper)

        def fuse_wrapper(original):
            def observed(feature, level, context, polar):
                self.record('feedback.fuse.'+level+'.confidence', context['confidence'], self.gate)
                return original(feature, level, context, polar)
            return observed
        self.patch_method(feedback, 'fuse', fuse_wrapper)

        def correct_wrapper(original):
            def observed(features, candidate, image, t1, context, polar):
                self.record('feedback.correct.confidence', context['confidence'], self.gate)
                return original(features, candidate, image, t1, context, polar)
            return observed
        self.patch_method(feedback, 'correct', correct_wrapper)

        for level, module in feedback.state.items():
            def state_hook(module, args, level=level):
                actual = args[0][:, -2:-1]
                expected = self.resized(self.gate, actual.shape[-2:])
                self.record('feedback.state.'+level+'.confidence', actual, expected)
                self.record('feedback.state.'+level+'.complement', args[0][:, -1:], 1-expected)
            self.handles.append(module.register_forward_pre_hook(state_hook))

        for level, module in feedback.decoder_inject.items():
            def decoder_hook(module, args, level=level):
                actual = args[0][:, -2:-1]
                expected = self.resized(self.gate, actual.shape[-2:])
                self.record('feedback.decoder_inject.'+level+'.confidence', actual, expected)
                self.record('feedback.decoder_inject.'+level+'.complement', args[0][:,-1:], 1-expected)
            self.handles.append(module.register_forward_pre_hook(decoder_hook))

        def output_hook(module, args):
            actual = args[0][:, -2:-1]
            expected = self.resized(self.gate, actual.shape[-2:])
            self.record('feedback.output_head.confidence', actual, expected)
            self.record('feedback.output_head.complement', args[0][:,-1:], 1-expected)
        self.handles.append(feedback.output_head.register_forward_pre_hook(output_hook))

        for level, module in self.model.r2p.adapters.items():
            def adapter_hook(module, args, level=level):
                actual = args[6]
                h, w = self.first.polar.shape[-2:]
                pad = (0, (-w) % 4, 0, (-h) % 4)
                expected = F.pad(self.gate, pad, mode='replicate') if any(pad) else self.gate
                self.record('enhancer.TaskAdapter.'+level+'.confidence', actual, expected)
            self.handles.append(module.register_forward_pre_hook(adapter_hook))

        guidance = self.model.error_guidance
        def guidance_hook(module, args):
            self.record('error_guidance.forward.error', args[3], self.b)
            self.record('error_guidance.forward.confidence', args[4], self.c)
            self.record('error_guidance.forward.context_confidence', args[6]['confidence'], self.gate)
        self.handles.append(guidance.register_forward_pre_hook(guidance_hook))

        def predict_hook(module, args):
            maps = args[0]
            self.record('error_guidance.predict.error_channel', maps[:,14:15],
                        self.resized(self.b, maps.shape[-2:]))
            self.record('error_guidance.predict.confidence_channel', maps[:,15:16],
                        self.resized(self.c, maps.shape[-2:]))
        self.handles.append(guidance.predict.register_forward_pre_hook(predict_hook))

        for level, module in guidance.inject.items():
            def inject_hook(module, args, level=level):
                actual = args[0][:,-1:]
                self.record('error_guidance.inject.'+level+'.confidence', actual,
                            self.resized(self.c, actual.shape[-2:]))
            self.handles.append(module.register_forward_pre_hook(inject_hook))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for owner, name, had_own_attribute, old_value in reversed(self.methods):
            if had_own_attribute:
                setattr(owner, name, old_value)
            else:
                delattr(owner, name)
        self.methods.clear()

    def result(self):
        required = ['feedback.make_context.confidence',
                    'feedback.correct.confidence',
                    'error_guidance.forward.error',
                    'error_guidance.forward.confidence',
                    'error_guidance.forward.context_confidence',
                    'error_guidance.predict.error_channel',
                    'error_guidance.predict.confidence_channel']
        required += ['feedback.state.level'+str(i)+'.confidence' for i in (1,2,3)]
        required += ['feedback.state.level'+str(i)+'.complement' for i in (1,2,3)]
        required += ['error_guidance.inject.level'+str(i)+'.confidence' for i in (1,2,3)]
        required += ['enhancer.TaskAdapter.level'+str(i)+'.confidence' for i in (2,3)]
        required += ['feedback.fuse.level'+str(i)+'.confidence' for i in (2,3)]
        required += ['feedback.decoder_inject.level'+str(i)+'.confidence' for i in (2,3)]
        required += ['feedback.decoder_inject.level'+str(i)+'.complement' for i in (2,3)]
        required += ['feedback.output_head.confidence', 'feedback.output_head.complement']
        seen = {event['path'] for event in self.events}
        missing = sorted(set(required)-seen) if self.enabled else []
        if missing:
            raise AssertionError(f'Quality path audit did not observe: {missing}')
        return dict(enabled=self.enabled, expected_paths=required,
                    complete=self.enabled and not missing, events=self.events,
                    coverage_count=len(seen), expected_coverage_count=len(required),
                    all_quality_paths_match=self.enabled and not missing and all(e['equal'] for e in self.events),
                    judgement_heads_executed=False, target_accessed=False)


def candidate_from_first(model, task, first, condition, fixed_b0=None, *,
                         capture_paths=True, study_module=None):
    """Generate U with all explicit quality paths replaced, preserving E2/task2.

    Each branch reuses the post-first RNG state and restores the caller's current
    RNG state on exit. The task2 sub-fork uses the original pre-task1 state. U is
    never clipped, accepted/rejected, or passed to M. This is not a training API.
    """
    _ensure_evaluation(model, task)
    if float(model.tau) != first.tau:
        raise ValueError('The inherited temperature changed after the shared first round.')
    if not hasattr(model, 'error_guidance'):
        raise ValueError('The declared Experiment 22 model must contain Q guidance.')
    study = _study(study_module)
    b_cond, c_cond, details = quality_conditions(first, condition, fixed_b0)
    # Retain the archived expression, including FP32 rounding, rather than gate=C.
    effective = 1. - 1. * (1. - c_cond)
    gate = F.interpolate(effective, size=first.polar.shape[-2:], mode='bilinear', align_corners=False)
    audit = _PathAudit(model, first, b_cond, c_cond, gate, capture_paths)
    try:
        audit.install()
        with _fork_from(first.candidate_rng_cpu, first.candidate_rng_cuda):
            context = model.feedback.make_context(first.image, first.t1, first.polar_summary,
                        dict(first.features), dict(first.enhanced.raw_features), gate)
            context, logits, need = model.error_guidance(first.image, first.t1,
                        first.polar_summary, b_cond, c_cond, first.features['T1'], context)
            # Original eval code removes the full-resolution task feature here.
            # Copy dictionaries so a later condition retains every first tensor.
            previous_features = dict(first.features)
            previous_features.pop('T1', None)
            second = study.enhance(model, first.polar, first.enhanced.refined_polar,
                        first.image, first.t1.clamp(0,1), previous_features, gate,
                        project=False, correction_state=context['state'])
            polar_context = model.feedback.polar_context(second.raw_features)
            second.raw_features.clear()
            with _fork_from(first.task_rng_cpu, first.task_rng_cuda):
                raw_candidate, task_features = study.task_forward(model, task,
                            second.refined_polar, first.image, first.polar,
                            context, polar_context)
            candidate, increment = model.feedback.correct(task_features, raw_candidate,
                            first.image, first.t1, context, polar_context)
    finally:
        audit.close()
    second.first = first.enhanced
    second.t1 = first.t1
    second.ungated_t2 = candidate
    second.candidate = candidate
    second.rgb_increment = increment
    second.first_task_features = dict(first.features)
    second.error_scale = first.error_scale
    second.confidence_map = first.confidence_map
    second.conditioned_error_scale = b_cond
    second.conditioned_confidence_map = c_cond
    second.effective_confidence = effective
    second.condition_gate = gate
    second.repair_logits = logits
    second.repair_need = need
    second.repair_threshold = model.good_threshold
    second.judgement_feature = task_features['T1']
    second.judgement_state = context['state']['level1']
    second.condition_details = details
    second.path_audit = audit.result()
    return second
