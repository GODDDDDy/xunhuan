# Actual frozen G6 adapter protocol

Only R2P trained, all TaskNet frozen; no alignment.
R2P pretrained e42 trunk with zero residual output head; initial identity.
Source G6 trained 3000 steps; additional adapter 3000 steps, fixed final checkpoint.
One actual forward; inherited round2 columns are aliases.
