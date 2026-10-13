"""Local need and candidate-benefit judgement; GT is used only for supervision."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def resize(x, size):
    return F.interpolate(x.float(), size=size, mode='bilinear', align_corners=False)


def local_error(prediction, target):
    """4x4 MAE plus edge MAE. Both inputs use the evaluation [0,1] range."""
    prediction, target = prediction.clamp(0, 1), target.clamp(0, 1)
    residual = (prediction - target).abs().mean(1, keepdim=True)
    dx = ((prediction[..., 1:] - prediction[..., :-1]) -
          (target[..., 1:] - target[..., :-1])).abs().mean(1, keepdim=True)
    dy = ((prediction[..., 1:, :] - prediction[..., :-1, :]) -
          (target[..., 1:, :] - target[..., :-1, :])).abs().mean(1, keepdim=True)
    edge = .5 * (F.pad(dx, (0, 1, 0, 0)) + F.pad(dy, (0, 0, 0, 1)))
    return F.avg_pool2d(residual + .25 * edge, 4)


def labels(t1, proposal, target, threshold, margin):
    with torch.no_grad():
        e1, ep = local_error(t1.detach(), target), local_error(proposal.detach(), target)
        improvement = e1 - ep
        # 0=better, 1=approximately unchanged, 2=worse. No test-derived labels.
        benefit = torch.ones_like(e1[:, 0], dtype=torch.long)
        benefit[improvement[:, 0] > margin] = 0
        benefit[improvement[:, 0] < -margin] = 2
        return (e1 > threshold).float(), benefit, e1, ep


class LocalAcceptance(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_projection = nn.Conv2d(96, 8, 1)
        self.body = nn.Sequential(nn.Conv2d(43, 32, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(32, 32, 3, padding=1, groups=32), nn.GELU(),
                                  nn.Conv2d(32, 4, 1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, image, polar, t1, proposal, error_scale, confidence, feature, state):
        # Gate parameters learn from their own supervision; these tensors cannot
        # carry comparator gradients into H, E, PolarFree or the candidate head.
        a, p = t1.detach().clamp(0, 1), proposal.detach().clamp(0, 1)
        maps = [F.avg_pool2d(x.detach().float(), 4)
                for x in (image, a, p, p-a, polar)]
        size = maps[0].shape[-2:]
        maps += [resize(error_scale.detach(), size), resize(confidence.detach(), size),
                 self.feature_projection(resize(feature.detach(), size)), resize(state.detach(), size)]
        logits = self.body(torch.cat(maps, 1))
        need = logits[:, :1].sigmoid()
        benefit = logits[:, 1:].softmax(1)
        better, same, worse = benefit.split(1, 1)
        allow = (need > .5) & (better > same) & (better > worse)
        weight = need * (better-worse).clamp_min(0)
        full_size = image.shape[-2:]
        # Interpolate weights but preserve an exact zero inside rejected cells.
        full_allow = F.interpolate(allow.float(), size=full_size, mode='nearest')
        mask = resize(weight, full_size) * full_allow
        return dict(logits=logits, need=need, benefit=benefit, mask=mask)

    @staticmethod
    def apply(t1, increment, judgement, step, training, warmup=500, transition=1000, override=None):
        predicted = judgement['mask'].detach()
        alpha = min(max((step-warmup)/transition, 0), 1) if training else 1.
        effective = (1-alpha) + alpha*predicted
        if override == 'mask_zero': effective = torch.zeros_like(effective)
        if override == 'mask_one': effective = torch.ones_like(effective)
        # torch.where makes the contract exact, including at inference boundaries.
        result = torch.where(effective == 0, t1, t1 + effective*increment)
        return result, effective


def judgement_loss(logits, need_target, benefit_target):
    need_loss = F.binary_cross_entropy_with_logits(logits[:, :1], need_target)
    # Fixed clipped inverse-frequency balancing within the current crop. Handles
    # initially all-neutral zero proposals without infinite or unstable weights.
    counts = torch.bincount(benefit_target.flatten(), minlength=3).float()
    weights = (counts.sum() / (3*counts.clamp_min(1))).clamp(.25, 4).detach()
    # Algebraically identical weighted cross entropy. PyTorch's CUDA spatial
    # NLL kernel has no deterministic implementation; preserve deterministic
    # training using one-hot multiplication and the same weighted denominator.
    one_hot = F.one_hot(benefit_target, num_classes=3).permute(0, 3, 1, 2).float()
    per_cell = -(one_hot * F.log_softmax(logits[:, 1:], dim=1)).sum(1)
    cell_weights = (one_hot * weights[None, :, None, None]).sum(1)
    benefit_loss = (per_cell * cell_weights).sum() / cell_weights.sum().clamp_min(1e-12)
    return need_loss + benefit_loss


class AcceptanceCollector:
    """Post-prediction checks; target never enters the forward or gate decision."""
    def __init__(self, root, split, threshold, margin):
        self.root, self.split, self.threshold, self.margin = root, split, threshold, margin
        self.rows, self.saved = [], set()
        self.totals = dict(cells=0, good=0, difficult=0, good_harmed=0, difficult_fixed=0,
                           accepted=0, accepted_harmful=0, useful_candidates=0,
                           rejected_useful=0, protected_output_pixels=0, output_pixels=0)
        self.confusion = np.zeros((3, 3), dtype=np.int64)

    def observe(self, out, batch):
        with torch.no_grad():
            need, target, e1, ep = labels(out.t1, out.ungated_t2, batch['target'], self.threshold, self.margin)
            ef = local_error(out.t2, batch['target'])
            mask = F.avg_pool2d(out.update_mask, 4)
            accepted = mask > 0
            good = need == 0
            harmful = ep > e1+self.margin
            useful = ep < e1-self.margin
            predicted = out.acceptance['benefit'].argmax(1)
            matrix = torch.bincount((target*3+predicted).flatten(), minlength=9).reshape(3, 3).cpu().numpy()
            self.confusion += matrix
            values = dict(cells=e1.numel(), good=int(good.sum()), difficult=int((~good).sum()),
                          good_harmed=int((good & (ef > e1+self.margin)).sum()),
                          difficult_fixed=int((~good & (ef < e1-self.margin)).sum()),
                          accepted=int(accepted.sum()), accepted_harmful=int((accepted & harmful).sum()),
                          useful_candidates=int(useful.sum()), rejected_useful=int((~accepted & useful).sum()),
                          protected_output_pixels=int((out.update_mask == 0).sum()), output_pixels=out.update_mask.numel())
            protected = (out.update_mask == 0).expand_as(out.t1)
            assert torch.equal(out.t1[protected], out.t2[protected])
            values.update(mask_mean=float(out.update_mask.mean()),
                          need_accuracy=float(((out.acceptance['need'] > .5) == (need > .5)).float().mean()),
                          proposal_error=float(ep.mean()), final_error=float(ef.mean()), t1_error=float(e1.mean()))
            for name in self.totals: self.totals[name] += values[name]
            self.rows.append(dict(sample_id=batch['sample_id'][0], scene=batch['scene'][0], **values))
            scene = batch['scene'][0]
            if scene not in self.saved:
                directory = self.root/'metrics/acceptance_examples'/self.split
                directory.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(directory/f'scene_{scene}.npz', sample_id=batch['sample_id'][0],
                    error1=e1[0,0].cpu().numpy(), proposal_error=ep[0,0].cpu().numpy(),
                    final_error=ef[0,0].cpu().numpy(), mask=mask[0,0].cpu().numpy(),
                    need=out.acceptance['need'][0,0].cpu().numpy(),
                    benefit=out.acceptance['benefit'][0].cpu().numpy())
                self.saved.add(scene)

    def finish(self):
        t = self.totals
        def ratio(a,b): return t[a]/t[b] if t[b] else None
        return dict(grid='4x4 local RGB MAE + .25 edge MAE', good_threshold=self.threshold,
                    benefit_margin=self.margin, totals=t, benefit_confusion=self.confusion.tolist(),
                    good_region_harm_rate=ratio('good_harmed','good'),
                    difficult_region_fix_rate=ratio('difficult_fixed','difficult'),
                    accepted_harmful_rate=ratio('accepted_harmful','accepted'),
                    rejected_useful_rate=ratio('rejected_useful','useful_candidates'),
                    protected_pixel_rate=ratio('protected_output_pixels','output_pixels'),
                    all_protected_pixels_exactly_preserved=True, per_image=self.rows,
                    target_used_only_after_prediction=True, calibration_not_from_test=True)
