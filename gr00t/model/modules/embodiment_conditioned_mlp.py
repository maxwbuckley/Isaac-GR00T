# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import torch.nn.functional as F


def swish(x):
    """Swish activation function."""
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


class CategorySpecificLinear(nn.Module):
    """Linear layer with category-specific weights and biases for multi-embodiment support."""

    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))
        # Lookup table mapping ORIGINAL category ids to rows of the (pruned)
        # weight tensors. None by default, which keeps the un-pruned fast path
        # zero-overhead (a single `is None` branch in forward). Set by
        # prune_categories(); entries of -1 mark ids that were pruned away.
        # Non-persistent: it never appears in state_dict, so un-pruned
        # checkpoints are byte-identical to before this feature existed.
        self.register_buffer("category_remap", None, persistent=False)

    def _resolve_cat_ids(self, cat_ids):
        """Map original category ids to pruned rows; raise on un-kept ids."""
        if self.category_remap is None:
            return cat_ids
        remapped = self.category_remap[cat_ids]
        if (remapped < 0).any():
            invalid = torch.unique(cat_ids[remapped < 0]).tolist()
            kept = torch.nonzero(self.category_remap >= 0).flatten().tolist()
            raise ValueError(
                f"This module was pruned to embodiment ids {kept} but received "
                f"un-kept embodiment ids {invalid}. Reload the model without "
                f"pruning (or prune with these ids included) to serve them."
            )
        return remapped

    def forward(self, x, cat_ids):
        """
        Args:
            x: [B, T, input_dim] input tensor
            cat_ids: [B] category/embodiment IDs
        Returns:
            [B, T, hidden_dim] output tensor
        """
        cat_ids = self._resolve_cat_ids(cat_ids)
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)

    def prune_categories(self, keep: list[int]) -> None:
        """Slice the per-category weights down to the ``keep`` ids, in place.

        This is a deployment-time, in-memory optimization: the [num_categories,
        in, out] weight/bias tensors are replaced with [len(keep), in, out]
        parameters holding the same values, and a remapping table is installed
        so subsequent forwards may keep using the ORIGINAL category ids.
        Forwards with an id not in ``keep`` raise a clear ValueError.

        A pruned module is NOT intended to be saved and re-loaded: its
        parameter shapes no longer match the checkpoint config. Prune after
        loading, never before saving.

        Args:
            keep: Original category ids to retain (non-empty, unique, each in
                [0, num_categories)).
        """
        if self.category_remap is not None:
            raise RuntimeError(
                "CategorySpecificLinear has already been pruned; pruning twice is not supported."
            )
        keep = [int(k) for k in keep]
        if not keep:
            raise ValueError("`keep` must contain at least one category id.")
        if len(set(keep)) != len(keep):
            raise ValueError(f"`keep` must not contain duplicate ids, got {keep}.")
        out_of_range = [k for k in keep if not 0 <= k < self.num_categories]
        if out_of_range:
            raise ValueError(
                f"Category ids {out_of_range} are out of range [0, {self.num_categories})."
            )
        device = self.W.device
        keep_index = torch.tensor(keep, dtype=torch.long, device=device)
        self.W = nn.Parameter(self.W.data[keep_index].clone(), requires_grad=self.W.requires_grad)
        self.b = nn.Parameter(self.b.data[keep_index].clone(), requires_grad=self.b.requires_grad)
        remap = torch.full((self.num_categories,), -1, dtype=torch.long, device=device)
        remap[keep_index] = torch.arange(len(keep), dtype=torch.long, device=device)
        self.category_remap = remap

    def expand_action_dimension(
        self, old_action_dim, new_action_dim, expand_input=False, expand_output=False
    ):
        """
        Safely expand action dimension with explicit targeting.

        Args:
            old_action_dim: Original action dimension
            new_action_dim: New (larger) action dimension
            expand_input: Whether to expand input dimension (dim=1)
            expand_output: Whether to expand output dimension (dim=2)
        """
        if new_action_dim <= old_action_dim:
            raise ValueError(
                f"New action dim {new_action_dim} must be larger than old action dim {old_action_dim}"
            )

        # Expand input dimension (dim=1) only if explicitly requested AND dimensions match
        if expand_input and self.W.shape[1] == old_action_dim:
            repeat_times = new_action_dim // old_action_dim
            remainder = new_action_dim % old_action_dim

            new_W_parts = [self.W] * repeat_times
            if remainder > 0:
                new_W_parts.append(self.W[:, :remainder, :])

            new_W = torch.cat(new_W_parts, dim=1)
            self.W = nn.Parameter(new_W)

        # Expand output dimension (dim=2) only if explicitly requested AND dimensions match
        if expand_output and self.W.shape[2] == old_action_dim:
            repeat_times = new_action_dim // old_action_dim
            remainder = new_action_dim % old_action_dim

            new_W_parts = [self.W] * repeat_times
            if remainder > 0:
                new_W_parts.append(self.W[:, :, :remainder])

            new_W = torch.cat(new_W_parts, dim=2)
            self.W = nn.Parameter(new_W)

            # Expand bias for output dimension
            if self.b.shape[1] == old_action_dim:
                new_b_parts = [self.b] * repeat_times
                if remainder > 0:
                    new_b_parts.append(self.b[:, :remainder])

                new_b = torch.cat(new_b_parts, dim=1)
                self.b = nn.Parameter(new_b)


class SmallMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        hidden = F.relu(self.layer1(x))
        return self.layer2(hidden)


class CategorySpecificMLP(nn.Module):
    """Two-layer MLP with category-specific weights for multi-embodiment support."""

    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        """
        Args:
            x: [B, T, input_dim] input tensor
            cat_ids: [B] category/embodiment IDs
        Returns:
            [B, T, output_dim] output tensor
        """
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)

    def prune_categories(self, keep: list[int]) -> None:
        """Prune both layers to the ``keep`` category ids.

        Deployment-time, in-memory optimization; a pruned module is NOT
        intended to be saved/re-loaded. See CategorySpecificLinear.prune_categories.
        """
        self.layer1.prune_categories(keep)
        self.layer2.prune_categories(keep)

    def expand_action_dimension(self, old_action_dim, new_action_dim):
        """
        Expand action dimension by copying weights from existing dimensions.

        Args:
            old_action_dim: Original action dimension
            new_action_dim: New (larger) action dimension
        """
        # self.layer1 does not take action_dim as input, so no expansion needed
        self.layer2.expand_action_dimension(
            old_action_dim, new_action_dim, expand_input=False, expand_output=True
        )


class MultiEmbodimentActionEncoder(nn.Module):
    """Action encoder with multi-embodiment support and sinusoidal positional encoding."""

    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        Args:
            actions: [B, T, action_dim] action tensor
            timesteps: [B,] timesteps - a single scalar per batch item
            cat_ids: [B,] category/embodiment IDs
        Returns:
            [B, T, hidden_size] encoded action features
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x

    def prune_categories(self, keep: list[int]) -> None:
        """Prune W1/W2/W3 to the ``keep`` embodiment ids.

        Deployment-time, in-memory optimization; a pruned module is NOT
        intended to be saved/re-loaded. See CategorySpecificLinear.prune_categories.
        """
        self.W1.prune_categories(keep)
        self.W2.prune_categories(keep)
        self.W3.prune_categories(keep)

    def expand_action_dimension(self, old_action_dim, new_action_dim):
        """
        Expand action dimension by copying weights from existing dimensions.

        Args:
            old_action_dim: Original action dimension
            new_action_dim: New (larger) action dimension
        """
        # Only W1 takes action_dim as input, so only expand its input dimension
        self.W1.expand_action_dimension(
            old_action_dim, new_action_dim, expand_input=True, expand_output=False
        )
