"""One temporary, differentiable alignment step; no live parameter update.

This is a preparation kernel, not a training controller.  Callers supply an
alignment loss on an inner TRAIN batch and a restoration/task objective on a
different outer TRAIN batch.  Both model states must expose a nonempty adaptable
subset.  The caller owns batch separation, objective definitions and optimizers.

All state tensors are snapshots.  Returned FANet hypergradients refer to its
snapshot tensors and must be explicitly copied to matching live parameters by a
future trainer.  Frozen weights are detached, but forwards are never wrapped in
no_grad: gradients through their INPUTS remain available.  Modules should already
be in the intended mode (normally eval for frozen backbones).  This file neither
changes modes nor RNG and does not approximate second-order derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Callable, Mapping, Optional, Sequence, Union

import torch
from torch import Tensor, nn
from torch.func import functional_call


LearningRates = Union[float, Mapping[str, float]]


def _unique_named(module: nn.Module, buffers: bool = False):
    source = module.named_buffers if buffers else module.named_parameters
    return dict(source(remove_duplicate=True))


def _aliases(module: nn.Module, buffers: bool = False):
    source = module.named_buffers if buffers else module.named_parameters
    canonical = {id(tensor): name for name, tensor in source(remove_duplicate=True)}
    return {name: canonical[id(tensor)] for name, tensor in source(remove_duplicate=False)}


def _canonical_state(module: nn.Module, supplied: Optional[Mapping[str, Tensor]],
                     *, buffers: bool = False):
    expected = _unique_named(module, buffers)
    if supplied is None:
        return expected
    aliases = _aliases(module, buffers)
    unknown = set(supplied) - set(aliases)
    if unknown:
        raise ValueError(f"Unknown {'buffer' if buffers else 'parameter'} names: {sorted(unknown)}")
    canonical = {}
    for name, value in supplied.items():
        key = aliases[name]
        if key in canonical and not torch.equal(canonical[key], value):
            raise ValueError(f"Conflicting tensors supplied for tied state {key!r}")
        canonical[key] = value
    if set(canonical) != set(expected):
        raise ValueError(f"Full state required; missing names: {sorted(set(expected) - set(canonical))}")
    for name, value in canonical.items():
        reference = expected[name]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"Shape/dtype mismatch at {name!r}")
    return canonical


@dataclass(frozen=True)
class FunctionalModelState:
    """A full canonical parameter/buffer snapshot with explicit trainable names.

    Tied aliases are canonicalized once, so one physical parameter is updated
    once.  ``functional_call(tie_weights=True)`` restores the sharing in forward.
    Buffer clones are passed on each call so BatchNorm or similar cannot mutate
    the fixed snapshot.  Supplied external state must include the complete model.
    """

    module: nn.Module
    parameters: Mapping[str, Tensor]
    buffers: Mapping[str, Tensor]
    trainable_names: tuple[str, ...]
    inner_lrs: Mapping[str, float]
    label: str = "model"
    _live_parameters: Mapping[str, Tensor] = field(default_factory=dict, repr=False)
    _live_buffers: Mapping[str, Tensor] = field(default_factory=dict, repr=False)

    @classmethod
    def from_module(cls, module: nn.Module, trainable_names: Sequence[str],
                    inner_lr: LearningRates = 1e-4, *, label: str = "model",
                    parameters: Optional[Mapping[str, Tensor]] = None,
                    buffers: Optional[Mapping[str, Tensor]] = None):
        aliases = _aliases(module)
        unknown = set(trainable_names) - set(aliases)
        if unknown:
            raise ValueError(f"{label}: unknown trainable names {sorted(unknown)}")
        names = tuple(aliases[name] for name in trainable_names)
        if len(set(names)) != len(names):
            raise ValueError(f"{label}: tied parameter selected more than once")
        if not names:
            raise ValueError(f"{label}: trainable subset must be nonempty")
        if isinstance(inner_lr, Mapping):
            rates = {aliases[name]: float(value) for name, value in inner_lr.items()}
            if set(rates) != set(names):
                raise ValueError(f"{label}: learning rates must match the trainable subset exactly")
        else:
            rates = {name: float(inner_lr) for name in names}
        if any(not math.isfinite(value) or value <= 0 for value in rates.values()):
            raise ValueError(f"{label}: inner learning rates must be finite and positive")
        fixed = _canonical_state(module, parameters)
        fixed_buffers = _canonical_state(module, buffers, buffers=True)
        snapshot = {}
        for name, value in fixed.items():
            if name in names and not (value.is_floating_point() or value.is_complex()):
                raise ValueError(f"{label}: selected parameter {name!r} is not differentiable")
            snapshot[name] = value.detach().clone().requires_grad_(name in names)
        return cls(module, snapshot,
                   {name: value.detach().clone() for name, value in fixed_buffers.items()},
                   names, rates, label,
                   {name: value.detach().clone() for name, value in _unique_named(module).items()},
                   {name: value.detach().clone() for name, value in _unique_named(module, True).items()})

    def call(self, *args, **kwargs):
        call_buffers = {name: value.clone() for name, value in self.buffers.items()}
        return functional_call(self.module, (dict(self.parameters), call_buffers),
                               args, kwargs, tie_weights=True, strict=True)

    def with_parameters(self, parameters: Mapping[str, Tensor]):
        if set(parameters) != set(self.parameters):
            raise ValueError(f"{self.label}: replacement must contain the full canonical parameter state")
        return replace(self, parameters=dict(parameters))

    def frozen(self):
        """Detach weights, preserving gradients from output back to input."""
        return replace(self, parameters={name: value.detach() for name, value in self.parameters.items()},
                       trainable_names=(), inner_lrs={})

    def assert_live_unchanged(self):
        for kind, expected, actual in (
            ("parameter", self._live_parameters, _unique_named(self.module)),
            ("buffer", self._live_buffers, _unique_named(self.module, True)),
        ):
            if set(expected) != set(actual):
                raise RuntimeError(f"{self.label}: live {kind} structure changed during probe")
            changed = [name for name, value in expected.items()
                       if not torch.equal(value, actual[name].detach())]
            if changed:
                raise RuntimeError(f"{self.label}: probe changed live {kind}s: {changed}")


Objective = Callable[[FunctionalModelState, FunctionalModelState, FunctionalModelState], Tensor]


def _scalar_loss(loss: Tensor, name: str):
    if not isinstance(loss, Tensor) or loss.numel() != 1:
        raise ValueError(f"{name} must return one scalar Tensor")
    if not bool(torch.isfinite(loss.detach()).all()):
        raise ValueError(f"{name} returned a nonfinite loss")
    return loss.reshape(())


def _selected(state: FunctionalModelState):
    if not state.trainable_names:
        raise ValueError(f"{state.label}: trainable subset must be nonempty")
    return tuple(state.parameters[name] for name in state.trainable_names)


def _checked_gradients(loss: Tensor, state: FunctionalModelState, *, create_graph: bool,
                       retain_graph: bool, require_all_connected: bool):
    if not loss.requires_grad:
        raise ValueError(f"{state.label}: objective has no gradient path to the selected subset")
    grads = torch.autograd.grad(loss, _selected(state), create_graph=create_graph,
                                retain_graph=retain_graph, allow_unused=True)
    unused = [name for name, grad in zip(state.trainable_names, grads) if grad is None]
    if len(unused) == len(grads) or (require_all_connected and unused):
        raise ValueError(f"{state.label}: objective does not use selected parameters {unused}")
    return {name: (grad if grad is not None else torch.zeros_like(state.parameters[name]))
            for name, grad in zip(state.trainable_names, grads)}


def _norm(grads: Mapping[str, Tensor]):
    return math.sqrt(sum(float(grad.detach().double().square().sum()) for grad in grads.values()))


@dataclass(frozen=True)
class BilevelProbeResult:
    inner_loss: Tensor
    outer_loss: Tensor
    temporary_upstream: FunctionalModelState
    temporary_downstream: FunctionalModelState
    inner_gradients: Mapping[str, Mapping[str, Tensor]]
    hypergradients: Mapping[str, Tensor]
    audit: Mapping[str, object]


def bilevel_probe(upstream: FunctionalModelState, downstream: FunctionalModelState,
                  alignment: FunctionalModelState, inner_alignment: Objective,
                  outer_objective: Objective, *, outer_create_graph: bool = True,
                  require_all_connected: bool = True) -> BilevelProbeResult:
    """A simultaneous one-step SGD probe on both model subsets.

    Inner loss must use the FANet state AND the selected upstream/downstream
    states.  Outer objective should assess restoration/task quality, normally
    without calling FANet directly.  ``create_graph=True`` in the inner derivative
    is mandatory: detaching that derivative destroys the learned-loss signal.
    No optimizer is invoked and no live ``.grad`` fields are populated.
    """
    _selected(upstream)
    _selected(downstream)
    _selected(alignment)
    inner = _scalar_loss(inner_alignment(upstream, downstream, alignment), "inner_alignment")
    up_grads = _checked_gradients(inner, upstream, create_graph=True, retain_graph=True,
                                 require_all_connected=require_all_connected)
    down_grads = _checked_gradients(inner, downstream, create_graph=True, retain_graph=True,
                                   require_all_connected=require_all_connected)
    up_parameters, down_parameters = dict(upstream.parameters), dict(downstream.parameters)
    for name, grad in up_grads.items():
        up_parameters[name] = upstream.parameters[name] - upstream.inner_lrs[name] * grad
    for name, grad in down_grads.items():
        down_parameters[name] = downstream.parameters[name] - downstream.inner_lrs[name] * grad
    temporary_up = upstream.with_parameters(up_parameters)
    temporary_down = downstream.with_parameters(down_parameters)
    outer = _scalar_loss(outer_objective(temporary_up, temporary_down, alignment), "outer_objective")
    hypergrads = _checked_gradients(outer, alignment, create_graph=outer_create_graph,
                                   retain_graph=True, require_all_connected=require_all_connected)
    for state in (upstream, downstream, alignment):
        state.assert_live_unchanged()
    audit = {"second_order": True, "optimizer_step_called": False,
             "live_parameters_unchanged": True, "live_buffers_unchanged": True,
             "upstream_trainable": list(upstream.trainable_names),
             "downstream_trainable": list(downstream.trainable_names),
             "alignment_trainable": list(alignment.trainable_names),
             "upstream_inner_lrs": dict(upstream.inner_lrs),
             "downstream_inner_lrs": dict(downstream.inner_lrs),
             "upstream_inner_gradient_norm": _norm(up_grads),
             "downstream_inner_gradient_norm": _norm(down_grads),
             "alignment_hypergradient_norm": _norm(hypergrads)}
    return BilevelProbeResult(inner, outer, temporary_up, temporary_down,
                             {"upstream": up_grads, "downstream": down_grads},
                             hypergrads, audit)


@dataclass(frozen=True)
class JointGradientResult:
    loss: Tensor
    upstream_gradients: Mapping[str, Tensor]
    downstream_gradients: Mapping[str, Tensor]
    audit: Mapping[str, object]


def joint_objective_gradients(upstream: FunctionalModelState, downstream: FunctionalModelState,
                             alignment: FunctionalModelState, objective: Objective, *,
                             create_graph: bool = False,
                             require_all_connected: bool = True) -> JointGradientResult:
    """Joint phase or matched baseline: fixed FANet, adaptive model subsets.

    For the collaboration phase, objective can combine alignment and task terms.
    For a matched baseline, objective omits alignment while selecting the exact
    same model parameter names.  FANet forwards retain gradients to model inputs.
    Again this function returns gradients only; the caller decides whether and
    when to apply them, outside preparation checks.
    """
    fixed_alignment = alignment.frozen()
    loss = _scalar_loss(objective(upstream, downstream, fixed_alignment), "joint_objective")
    up_grads = _checked_gradients(loss, upstream, create_graph=create_graph, retain_graph=True,
                                 require_all_connected=require_all_connected)
    down_grads = _checked_gradients(loss, downstream, create_graph=create_graph, retain_graph=True,
                                   require_all_connected=require_all_connected)
    for state in (upstream, downstream, alignment):
        state.assert_live_unchanged()
    return JointGradientResult(loss, up_grads, down_grads,
                               {"alignment_frozen": True, "optimizer_step_called": False,
                                "live_parameters_unchanged": True, "live_buffers_unchanged": True,
                                "upstream_trainable": list(upstream.trainable_names),
                                "downstream_trainable": list(downstream.trainable_names),
                                "upstream_gradient_norm": _norm(up_grads),
                                "downstream_gradient_norm": _norm(down_grads)})


# Descriptive alias for callers whose joint objective is pure feature alignment.
joint_alignment_gradients = joint_objective_gradients
