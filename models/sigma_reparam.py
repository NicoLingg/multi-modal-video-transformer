# Implementation of Sigma Reparameterization for Spectral Normalization: https://arxiv.org/abs/2303.06296
# Based on Apple implemenation: https://github.com/apple/ml-sigma-reparam/tree/main

"""SigmaReparam Layers."""

import functools
import inspect
import typing as t
import torch
from torch import nn
import torch.nn.functional as F


class SpectralNormedWeight(nn.Module):
    """SpectralNorm Layer. First sigma uses SVD, then power iteration."""

    def __init__(
        self,
        weight: torch.Tensor,
    ):
        super().__init__()
        self.weight = weight
        with torch.no_grad():
            _, s, vh = torch.linalg.svd(self.weight, full_matrices=False)

        self.register_buffer("u", vh[0])
        self.register_buffer("spectral_norm", s[0] * torch.ones(1))

    def get_sigma(self, u: torch.Tensor, weight: torch.Tensor):
        with torch.no_grad():
            v = weight.mv(u)
            v = nn.functional.normalize(v, dim=0)
            u = weight.T.mv(v)
            u = nn.functional.normalize(u, dim=0)
            if self.training:
                self.u.data.copy_(u)

        return torch.einsum("c,cd,d->", v, weight, u)

    def forward(self):
        """Normalize by largest singular value and rescale by learnable."""
        sigma = self.get_sigma(u=self.u, weight=self.weight)
        if self.training:
            self.spectral_norm.data.copy_(sigma)

        return self.weight / sigma


class SNLinear(nn.Linear):
    """Spectral Norm linear from sigmaReparam.

    Optionally, if 'stats_only' is `True`,then we
    only compute the spectral norm for tracking
    purposes, but do not use it in the forward pass.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_multiplier: float = 1.0,
        stats_only: bool = False,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.stats_only = stats_only
        self.init_multiplier = init_multiplier

        self.init_std = 0.02 * init_multiplier
        nn.init.trunc_normal_(self.weight, std=self.init_std)

        # Handle normalization and add a learnable scalar.
        self.spectral_normed_weight = SpectralNormedWeight(self.weight)
        sn_init = self.spectral_normed_weight.spectral_norm

        # Would have set sigma to None if `stats_only` but jit really disliked this
        self.sigma = (
            torch.ones_like(sn_init)
            if self.stats_only
            else nn.Parameter(
                torch.zeros_like(sn_init).copy_(sn_init), requires_grad=True
            )
        )

        self.register_buffer("effective_spectral_norm", sn_init)
        self.update_effective_spec_norm()

    def update_effective_spec_norm(self):
        """Update the buffer corresponding to the spectral norm for tracking."""
        with torch.no_grad():
            s_0 = (
                self.spectral_normed_weight.spectral_norm
                if self.stats_only
                else self.sigma
            )
            self.effective_spectral_norm.data.copy_(s_0)

    def get_weight(self):
        """Get the reparameterized or reparameterized weight matrix depending on mode
        and update the external spectral norm tracker."""
        normed_weight = self.spectral_normed_weight()
        self.update_effective_spec_norm()
        return self.weight if self.stats_only else normed_weight * self.sigma

    def forward(self, inputs: torch.Tensor):
        weight = self.get_weight()
        return F.linear(inputs, weight, self.bias)


def convert_layers(
    container: nn.Module,
    from_layer: t.Callable = nn.Linear,
    to_layer: t.Callable = SNLinear,
    set_from_layer_kwargs: bool = True,
    ignore_from_signature: t.Optional[t.Iterable] = ("device", "dtype"),
) -> nn.Module:
    """Convert from_layer to to_layer for all layers in container."""
    for child_name, child in container.named_children():
        if isinstance(child, from_layer):
            to_layer_i = to_layer
            if set_from_layer_kwargs:
                signature_list = inspect.getfullargspec(from_layer).args[
                    1:
                ]  # 0th element is arg-list, 0th of that is 'self'
                if ignore_from_signature is not None:
                    signature_list = [
                        k for k in signature_list if k not in ignore_from_signature
                    ]

                kwargs = {
                    sig: (
                        getattr(child, sig)
                        if sig != "bias"
                        else bool(child.bias is not None)
                    )
                    for sig in signature_list
                }
                to_layer_i = functools.partial(to_layer, **kwargs)

            setattr(container, child_name, to_layer_i())
        else:
            convert_layers(
                child,
                from_layer,
                to_layer,
                set_from_layer_kwargs=set_from_layer_kwargs,
                ignore_from_signature=ignore_from_signature,
            )

    return container
