from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from model import ModelConfig
from utils import collect_by_leaf_name, path_to_str


# ==========================================
# Muon (Newton-Schulz orthogonalization)
# ==========================================
def muon_orthogonalize(w, g, lr, ns_steps: int = 5):
    """Orthogonalize the gradient via Newton-Schulz iteration, then take a step.

    Fixes vs. the original version:
      - normalizes the input to the iteration (required for NS to converge to an
        orthogonal factor instead of collapsing towards 0 or diverging)
      - applies `lr` AFTER orthogonalization, not before
      - supports 3D expert-stacked weights (num_experts, in, out) via batched matmuls
    """
    eps = 1e-7
    if w.ndim == 3:
        norm = jnp.linalg.norm(g, axis=(-2, -1), keepdims=True)
        X = g / (norm + eps)
        for _ in range(ns_steps):
            X = 1.5 * X - 0.5 * jnp.einsum("eij,ejk,ekl->eil", X, jnp.swapaxes(X, -1, -2), X)
    else:
        norm = jnp.linalg.norm(g)
        X = g / (norm + eps)
        for _ in range(ns_steps):
            X = 1.5 * X - 0.5 * X @ X.T @ X

    return w - (X * lr)


class MuonState(NamedTuple):
    count: jnp.ndarray


def make_hybrid_optimizer(total_steps: int):
    # Warmup (first 5% of steps) + cosine decay, shared across all three sub-optimizers.
    warmup_steps = max(1, int(total_steps * 0.05))
    cosine = optax.cosine_decay_schedule(
        init_value=1.0, decay_steps=max(1, total_steps - warmup_steps), alpha=0.1
    )
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(init_value=0.0, end_value=1.0, transition_steps=warmup_steps),
            cosine,
        ],
        boundaries=[warmup_steps],
    )

    lion_lr = lambda step: 3e-4 * lr_schedule(step)
    adamw_lr = lambda step: 1e-3 * lr_schedule(step)
    tx_lion = optax.lion(learning_rate=lion_lr, weight_decay=0.1)
    # Anti-overfitting: the embedding table is the single largest weight matrix in the
    # model and, unlike everything else that used to be lumped into "adamw" (norm
    # scales, biases), it's exactly the kind of large lookup table that CAN overfit --
    # rare tokens get few gradient updates and their rows can drift/memorize. Give it a
    # small decay of its own instead of the previous universal weight_decay=0.0.
    tx_adamw_decay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.01)
    tx_adamw_nodecay = optax.adamw(learning_rate=adamw_lr, weight_decay=0.0)

    def _muon_step(base_lr: float):
        def init_fn(params):
            return MuonState(count=jnp.zeros([], jnp.int32))

        def update_fn(updates, state, params=None):
            if params is None:
                return updates, state
            step_lr = base_lr * lr_schedule(state.count)
            new_updates = jax.tree_util.tree_map(
                lambda p, g: (muon_orthogonalize(p, g, step_lr) - p), params, updates
            )
            return new_updates, MuonState(count=state.count + 1)

        return optax.GradientTransformation(init_fn, update_fn)

    tx_muon = _muon_step(base_lr=0.02)

    def _label_leaf(path, param):
        path_str = path_to_str(path)
        # Exclude huge embedding/lm_head matrices and 1D params from Muon: Newton-Schulz on a
        # (vocab_size, d_model) matrix would compute an X @ X.T of size vocab_size^2 -> OOM.
        if "embed" in path_str or "lm_head" in path_str:
            return "adamw_decay"
        if "rmsnorm" in path_str or "bias" in path_str:
            return "adamw_nodecay"
        if param.ndim >= 2:
            if "mamba" in path_str:
                return "lion"
            return "muon"  # includes 3D expert-stacked weights -> handled by batched Muon branch
        return "lion"

    def label_fn(params):
        
        return jax.tree_util.tree_map_with_path(_label_leaf, params)

    clip_tx = optax.clip_by_global_norm(1.0)
    multi_tx = optax.multi_transform(
        {"muon": tx_muon, "lion": tx_lion, "adamw_decay": tx_adamw_decay, "adamw_nodecay": tx_adamw_nodecay},
        label_fn,
    )
    return optax.chain(clip_tx, multi_tx)


# ==========================================
# Loss
# ==========================================
def compute_loss(params, model_fn, batch, cfg: ModelConfig, rngs=None, deterministic=False, return_aux=False):
    input_ids = batch["input_ids"]
    labels = batch["labels"]

    kwargs = {"deterministic": deterministic}
    if rngs is not None:
        kwargs["rngs"] = rngs

    outputs = model_fn(
        {"params": params}, input_ids, **kwargs, mutable=["losses"] if not deterministic else False
    )

    expert_util_stacked = None
    if not deterministic:
        logits, sowed_vars = outputs
        # FIX: sow() collections nest by module scope (one entry per layer's "moe"
        # submodule), not a flat {"aux_loss": (...)} at the top -- see
        # collect_by_leaf_name's docstring. jnp.array(sowed_vars["losses"]["aux_loss"])
        # would raise KeyError the moment this ever ran against real Flax.
        aux_losses = collect_by_leaf_name(sowed_vars["losses"], "aux_loss")
        z_losses = collect_by_leaf_name(sowed_vars["losses"], "z_loss")
        expert_utils = collect_by_leaf_name(sowed_vars["losses"], "expert_utilization")
        aux_loss = jnp.sum(jnp.stack(aux_losses)) if aux_losses else 0.0
        z_loss = jnp.sum(jnp.stack(z_losses)) if z_losses else 0.0
        if expert_utils:
            expert_util_stacked = jnp.stack(expert_utils)  # (num_layers, num_experts)
    else:
        logits = outputs
        aux_loss, z_loss = 0.0, 0.0

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    vocab_size = logits.shape[-1]
    one_hot_labels = jax.nn.one_hot(labels, num_classes=vocab_size)

    # Anti-overfitting: label smoothing softens the one-hot target so the model isn't
    # pushed to drive the correct-token probability all the way to 1 -- standard
    # regularizer for large-vocabulary LMs. Set cfg.label_smoothing=0.0 to disable.
    if cfg.label_smoothing > 0:
        smooth_positive = 1.0 - cfg.label_smoothing
        smooth_negative = cfg.label_smoothing / (vocab_size - 1)
        one_hot_labels = one_hot_labels * smooth_positive + smooth_negative * (1.0 - one_hot_labels)

    loss_matrix = -jnp.sum(log_probs * one_hot_labels, axis=-1)
    mask = (labels != -100).astype(jnp.float32)
    ce_loss = jnp.sum(loss_matrix * mask) / (jnp.sum(mask) + 1e-9)

    total_loss = ce_loss + (cfg.router_aux_loss_coef * aux_loss) + (cfg.router_z_loss_coef * z_loss)
    if return_aux:
        aux_info = {
            "ce_loss": ce_loss,
            "aux_loss": aux_loss,
            "z_loss": z_loss,
            "expert_utilization": expert_util_stacked,  # (num_layers, num_experts) or None in eval mode
        }
        return total_loss, aux_info
    return total_loss
