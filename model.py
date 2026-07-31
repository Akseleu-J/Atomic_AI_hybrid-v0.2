import math

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from typing import List
from jax.sharding import PartitionSpec as P
from jax.experimental.pallas.ops.tpu.flash_attention import (
    flash_attention as pallas_flash_attention,
    BlockSizes as FlashBlockSizes,
)


@struct.dataclass
class ModelConfig:
    d_model: int = 1024
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 16
    d_latent: int = 256
    d_ff: int = 3072
    num_experts: int = 8
    top_k: int = 2
    num_layers: int = 22
    vocab_size: int = 151936
    dropout_rate: float = 0.1
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.0001
    moe_capacity_factor: float = 1.25
    tie_embeddings: bool = True
    label_smoothing: float = 0.05
    router_noise_std: float = 0.3
    # ЧЕМ ВЫЗВАН OOM НА MLA: наивная attention материализует score-матрицу формы
    # (b, n_heads, l, l). При l=8192, n_heads=16 это b*16*8192*8192 элементов --
    # ~2.15 ГБ на bf16 ТОЛЬКО под сам score-тензор, ОДИН слой, при b=1 (локальный
    # батч на чип после data-parallel шардирования на 8 TPU). Softmax, causal
    # mask (jnp.where создаёт новый массив) и dropout-маска добавляют ещё 2-3
    # временных тензора того же размера поверх -- итого 6-9 ГБ пиковой памяти на
    # ОДИН слой ещё до учёта весов/оптимизатора. remat не спасает: он убирает
    # накопление активаций МЕЖДУ слоями, но не уменьшает размер активации ВНУТРИ
    # одного слоя. Дальше N^2 не убрать без смены алгоритма -- отсюда Pallas
    # Flash Attention: kernel считает attention блоками (block_q x block_k) и
    # никогда не материализует полную (l, l) матрицу в HBM.
    use_flash_attention: bool = True
    # Размер чанка для chunked-scan в GatedDeltaNet2 (см. комментарий в
    # GatedDeltaNet2J). Должен делить seq_len нацело. 8192 делится на 512, 1024,
    # 2048 и т.д. -- выбирай по тому, сколько памяти готов отдать под один чанк
    # (chunk_size * n_heads * d_head^2 * 2 байта на M/C/P каждый) против числа
    # последовательных шагов scan (l / chunk_size).
    deltanet_chunk_size: int = 1024


# ==========================================
# RoPE
# ==========================================
class RoPEEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, seq_len):
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2)[: (self.dim // 2)] / self.dim))
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("i,j->ij", t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x, cos, sin):
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    rotated_x = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated_x * sin


# ==========================================
# Multi-head Latent Attention
# ==========================================
#
# ВАЖНО про dropout: Pallas flash_attention НЕ поддерживает dropout внутри
# кернела (нет такого параметра в его сигнатуре -- проверено по исходнику
# jax/experimental/pallas/ops/tpu/flash_attention.py). Поэтому dropout на
# attention-весах убран из flash-пути; регуляризация остаётся за счёт dropout
# в FFN/MoE (ExpertPack) и label smoothing -- это стандартная практика (сам
# GPT-3/LLaMA-класс моделей обычно не использует attention dropout при
# длинных контекстах именно по этой причине). Наивный путь (deterministic
# отладка на CPU/маленьких seq_len, где Pallas TPU-кернел недоступен) дропаут
# сохраняет.
class MLAJ(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, causal_mask, cos, sin, deterministic: bool = True, rngs=None):
        mesh = rngs.get("mesh") if rngs is not None else None
        b, l, _ = x.shape
        n_heads = self.cfg.n_heads
        d_head = self.cfg.d_model // n_heads

        Q = nn.Dense(self.cfg.d_model, use_bias=False, name="W_q")(x)
        Q = Q.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)  # (b, n_heads, l, d_head)
        Q_rope = apply_rope(Q, cos[None, None, :, :d_head], sin[None, None, :, :d_head])

        kv_latent = nn.Dense(self.cfg.d_latent, use_bias=False, name="W_kv_down")(x)
        K = nn.Dense(self.cfg.d_model, use_bias=False, name="W_k_up")(kv_latent)
        V = nn.Dense(self.cfg.d_model, use_bias=False, name="W_v_up")(kv_latent)

        K = K.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)
        K_rope = apply_rope(K, cos[None, None, :, :d_head], sin[None, None, :, :d_head])
        V = V.reshape(b, l, n_heads, d_head).transpose(0, 2, 1, 3)

        # sm_scale must be a concrete Python float -- it is a static_argname in
        # pallas_flash_attention's jax.jit signature, so a jnp-traced value here
        # breaks under remat's retracing (float(tracer) raises
        # ConcretizationTypeError). d_head is already a plain Python int.
        sm_scale = 1.0 / math.sqrt(d_head)

        if self.cfg.use_flash_attention:
            # ЧТО ЗНАЧИТ ЭТА ОШИБКА: "Mosaic kernels cannot be automatically
            # partitioned. Please wrap the call in a shard_map." Pallas/Mosaic
            # kernели -- это opaque custom-call для XLA: GSPMD (jax.jit +
            # in_shardings) не умеет заглянуть внутрь и решить, как разбить его
            # между чипами, в отличие от обычных jnp-операций (einsum, cumsum и
            # т.д.), для которых GSPMD ищет разбиение сам. Явный shard_map --
            # единственный способ сказать "вот так шардировать", а не "прикинь
            # сам".
            #
            # Здесь шардируем ТОЛЬКО batch-ось (op "tpu_nodes") -- каждый чип
            # запускает kernel на своём локальном шарде батча, всё остальное
            # (n_heads, l, d_head) реплицировано, что совпадает с тем, как уже
            # шардированы данные в train.py (data_sharding = P("tpu_nodes", None)).
            #
            # ВАЖНО: block_sizes должен считаться от ЛОКАЛЬНОГО batch (внутри
            # shard_map), а не от глобального `b` из внешней области видимости --
            # kernel реально увидит только свой локальный шард. Проверено
            # отдельно на CPU (4 симулированных устройства): функция внутри
            # shard_map получает shape с локальным batch=1 при глобальном
            # batch=4, а не глобальный размер.
            #
            # mesh берётся из ambient jax.set_mesh(mesh), который уже
            # используется в train.py вокруг тренировочного цикла -- явно
            # передавать mesh в модель не нужно.
            def _flash_call(q_local, k_local, v_local):
                local_b = q_local.shape[0]
                block_sizes = FlashBlockSizes.get_default(local_b, n_heads, l, l, d_head)
                return pallas_flash_attention(
                    q_local, k_local, v_local,
                    causal=True, sm_scale=sm_scale, block_sizes=block_sizes,
                )
            if self.cfg.use_flash_attention and mesh is not None:
                sharded_flash = jax.shard_map(
                    _flash_call,
                    mesh=mesh,
                    in_specs=P("tpu_nodes", None, None, None),
                    out_specs=P("tpu_nodes", None, None, None),
                    check_vma=False,
                )
                out = sharded_flash(
                        Q_rope.astype(jnp.bfloat16), K_rope.astype(jnp.bfloat16), V.astype(jnp.bfloat16)
                    ).astype(x.dtype)
            else:
                out = _flash_call(
                    Q_rope.astype(jnp.bfloat16),
                    K_rope.astype(jnp.bfloat16),
                    V.astype(jnp.bfloat16)
                ).astype(x.dtype)
                
        else:
            # Naive fallback -- only for CPU debugging / small seq_len smoke tests.
            # This is the O(l^2) path that caused the MLA OOM at seq_len=8192.
            scores = jnp.einsum("bhqd,bhkd->bhqk", Q_rope, K_rope) * sm_scale
            scores = jnp.where(causal_mask == 0, -1e9, scores)
            attn = jax.nn.softmax(scores, axis=-1)
            if not deterministic:
                dropout_rng = rngs['dropout'] if rngs is not None and 'dropout' in rngs else self.make_rng('dropout')
                keep_prob = 1.0 - self.cfg.dropout_rate
                mask_drop = jax.random.bernoulli(dropout_rng, keep_prob, attn.shape)
                attn = attn * mask_drop / keep_prob
            out = jnp.einsum("bhqk,bhkd->bhqd", attn, V)

        out = out.transpose(0, 2, 1, 3).reshape(b, l, self.cfg.d_model)
        return nn.Dense(self.cfg.d_model, use_bias=False, name="W_o")(out)


# ==========================================
# Mamba-2 (SSM)
# ==========================================
class Mamba2J(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        b, l, d = x.shape
        d_inner = d * self.cfg.expand

        in_proj = nn.Dense(d_inner * 2, use_bias=False, name="in_proj")(x)
        x_bc, res = jnp.split(in_proj, 2, axis=-1)

        conv_w = self.param("conv_w", nn.initializers.normal(stddev=0.02), (d_inner, self.cfg.d_conv))
        conv_b = self.param("conv_b", nn.initializers.zeros, (d_inner,))

        rhs = conv_w.T[:, None, :]
        res_conv = jax.lax.conv_general_dilated(
            lhs=x_bc,
            rhs=rhs,
            window_strides=(1,),
            padding=[(self.cfg.d_conv - 1, 0)],
            feature_group_count=d_inner,
            dimension_numbers=('NHC', 'HIO', 'NHC')
        )
        x_conv = jax.nn.silu(res_conv + conv_b[None, None, :])

        A = -jnp.exp(self.param("A_log", nn.initializers.uniform(scale=1.0), (d_inner,)))
        B = nn.Dense(self.cfg.d_state, use_bias=False, name="B_proj")(x_bc)
        C = nn.Dense(self.cfg.d_state, use_bias=False, name="C_proj")(x_bc)
        dt = jax.nn.softplus(nn.Dense(d_inner, use_bias=True, name="dt_proj")(x_bc))

        dA = jnp.exp(jnp.einsum("bld,d->bld", dt, A))
        dB = jnp.einsum("bld,bls->blds", dt, B)

        def _associative_scan_mamba(a, b_val):
            da1, db1, xc1 = a
            da2, db2, xc2 = b_val
            da2e = da2[..., None]
            return da2 * da1, da2e * db1 + db2, da2e * xc1 + xc2

        _, _, h = jax.lax.associative_scan(
            _associative_scan_mamba, (dA, dB, x_conv[..., None]), axis=1
        )
        y = jnp.einsum("blds,bls->bld", h, C)
        out = y * jax.nn.silu(res)
        return nn.Dense(d, use_bias=False, name="out_proj")(out)


# ==========================================
# Gated DeltaNet-2
# ==========================================
class GatedDeltaNet2J(nn.Module):
    """Gated Delta Rule-2 (Hatamizadeh, Choi, Kautz -- NVIDIA, arXiv:2605.22791, May 2026)."""

    cfg: ModelConfig

    @nn.compact
    def __call__(self, x):
        b, l, d = x.shape
        n_heads = self.cfg.n_heads
        d_head = d // n_heads
        eps = 1e-6

        def short_causal_conv(name, u):
            conv_w = self.param(f"{name}_conv_w", nn.initializers.normal(stddev=0.02), (d, self.cfg.d_conv))
            conv_b = self.param(f"{name}_conv_b", nn.initializers.zeros, (d,))
            rhs = conv_w.T[:, None, :]
            out = jax.lax.conv_general_dilated(
                lhs=u,
                rhs=rhs,
                window_strides=(1,),
                padding=[(self.cfg.d_conv - 1, 0)],
                feature_group_count=d,
                dimension_numbers=('NHC', 'HIO', 'NHC')
            )
            return out + conv_b[None, None, :]

        q_lin = nn.Dense(d, use_bias=False, name="q_proj")(x)
        k_lin = nn.Dense(d, use_bias=False, name="k_proj")(x)
        v_lin = nn.Dense(d, use_bias=False, name="v_proj")(x)

        q = jax.nn.silu(short_causal_conv("q", q_lin)).reshape(b, l, n_heads, d_head)
        k = jax.nn.silu(short_causal_conv("k", k_lin)).reshape(b, l, n_heads, d_head)
        v = jax.nn.silu(short_causal_conv("v", v_lin)).reshape(b, l, n_heads, d_head)

        q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + eps)
        k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + eps)

        b_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, name="erase_gate")(x)).reshape(b, l, n_heads, d_head)
        w_gate = jax.nn.sigmoid(nn.Dense(d, use_bias=True, name="write_gate")(x)).reshape(b, l, n_heads, d_head)

        a_param = self.param("decay_a", nn.initializers.zeros, (n_heads,))
        f_proj = nn.Dense(d, use_bias=True, name="decay_proj")(x).reshape(b, l, n_heads, d_head)
        g = -jnp.exp(a_param)[None, None, :, None] * jax.nn.softplus(f_proj)
        alpha = jnp.exp(g)

        out_gate = nn.Dense(d, use_bias=False, name="out_gate")(x)

        e = b_gate * k
        z = w_gate * v
        ea = e * alpha

        # ---- chunked delta-rule scan ----
        # ЧТО БЫЛО: M и C строились для ВСЕЙ последовательности разом --
        # (b, l, n_heads, d_head, d_head). При l=8192, n_heads=16, d_head=64 это
        # ~1.07 ГБ НА КАЖДЫЙ из M и C (bf16) -- ещё до самого associative_scan,
        # который дополнительно требует сопоставимый объём для промежуточных
        # уровней дерева. Это отдельный источник давления на HBM, независимый от
        # MoE/MLA фиксов выше.
        #
        # ЧТО СТАЛО: последовательность бьётся на чанки по оси времени; вместо
        # ОДНОГО associative_scan на всю длину -- jax.lax.scan ПО ЧАНКАМ
        # (последовательно), а ВНУТРИ каждого чанка -- свой маленький
        # associative_scan (параллельный, но только на chunk_size шагов). M_c/C_c
        # строятся ЗАНОВО из k/ea/z/alpha (эти малы -- O(l*d), а не O(l*d^2))
        # только для текущего чанка внутри scan, поэтому пиковая память для
        # M/C/P -- O(chunk_size * d_head^2), а не O(l * d_head^2).
        #
        # ВАЖНО про корректность: между чанками нужно пронести НЕ ТОЛЬКО
        # состояние S (значение), но и накопленное произведение матриц M
        # (назовём carry_M) -- это ровно то, что associative_scan обычно
        # выбрасывает (`_, S = ...`). Без carry_M нельзя корректно доклеить
        # следующий чанк: S_global_i (внутри чанка) = P_local_i @ carry_M +
        # (P_local_i @ carry_S + S_local_i), что в точности совпадает с
        # применением того же самого `_combine` к паре (carry_M, carry_S) и
        # локальному (P_local_i, S_local_i) -- т.е. это не новая математика, а
        # использование ассоциативности того же самого combine-оператора.
        # Проверено численно (см. чат): расхождение с нечанкованной версией на
        # уровне float32-округления (~1e-6), не более.
        chunk_size = min(self.cfg.deltanet_chunk_size, l)
        if l % chunk_size != 0:
            raise ValueError(
                f"seq_len={l} must be divisible by deltanet_chunk_size={chunk_size} "
                "(chunked GatedDeltaNet2 scan requires equal-sized chunks)."
            )
        num_chunks = l // chunk_size

        def _combine(state1, state2):
            m1, c1 = state1
            m2, c2 = state2
            return m2 @ m1, m2 @ c1 + c2

        def _to_chunks(t):  # (b, l, h, d) -> (num_chunks, b, chunk_size, h, d)
            t = t.reshape(b, num_chunks, chunk_size, n_heads, d_head)
            return jnp.moveaxis(t, 1, 0)

        k_ch, ea_ch, z_ch, alpha_ch, q_ch = map(_to_chunks, (k, ea, z, alpha, q))

        eye_bh = jnp.broadcast_to(jnp.eye(d_head, dtype=x.dtype), (b, n_heads, d_head, d_head))
        zero_bh = jnp.zeros((b, n_heads, d_head, d_head), dtype=x.dtype)

        def _chunk_step(carry, chunk_inputs):
            carry_M, carry_S = carry
            k_c, ea_c, z_c, alpha_c, q_c = chunk_inputs  # each (b, chunk_size, h, d)

            eye = jnp.eye(d_head, dtype=x.dtype)[None, None, None, :, :]
            M_c = eye * alpha_c[:, :, :, None, :] - k_c[:, :, :, :, None] @ ea_c[:, :, :, None, :]
            C_c = k_c[:, :, :, :, None] @ z_c[:, :, :, None, :]

            P_local, S_local = jax.lax.associative_scan(_combine, (M_c, C_c), axis=1)

            # broadcast carry (no chunk-position axis) against P_local's chunk axis
            global_M = jnp.einsum("bchmn,bhnp->bchmp", P_local, carry_M)
            global_S = jnp.einsum("bchmn,bhnp->bchmp", P_local, carry_S) + S_local

            out_c = jnp.einsum("bchij,bchi->bchj", global_S, q_c)
            new_carry = (global_M[:, -1], global_S[:, -1])
            return new_carry, out_c

        _, out_chunks = jax.lax.scan(
            _chunk_step, (eye_bh, zero_bh), (k_ch, ea_ch, z_ch, alpha_ch, q_ch)
        )
        out = jnp.moveaxis(out_chunks, 0, 1).reshape(b, l, d)  # (num_chunks,b,c,h,d) -> (b,l,d)

        out = nn.RMSNorm(epsilon=1e-6, name="out_norm")(out)
        return nn.Dense(d, use_bias=False, name="out_proj")(out * jax.nn.silu(out_gate))


# ==========================================
# MoE -- gather/scatter dispatch, O(N) instead of O(N * capacity)
# ==========================================
class ExpertPack(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        h = nn.Dense(self.cfg.d_ff, name="w1")(x)
        h = jax.nn.gelu(h)
        h = nn.Dropout(rate=self.cfg.dropout_rate)(h, deterministic=deterministic)
        return nn.Dense(self.cfg.d_model, name="w2")(h)


class MoEJ(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic: bool = True, rngs=None):
        b, l, d = x.shape
        flat_x = x.reshape(-1, d)
        num_tokens = flat_x.shape[0]
        E, K = self.cfg.num_experts, self.cfg.top_k
        n_assign = num_tokens * K

        router_logits = nn.Dense(E, use_bias=False, name="router")(flat_x)
        if not deterministic and self.cfg.router_noise_std > 0:
            noise_rng = self.make_rng("dropout")
            router_logits = router_logits + self.cfg.router_noise_std * jax.random.normal(
                noise_rng, router_logits.shape
            )

        router_probs = jax.nn.softmax(router_logits, axis=-1)
        top_k_vals, top_k_idx = jax.lax.top_k(router_probs, K)  # (num_tokens, K)

        gate = top_k_vals / (jnp.sum(top_k_vals, axis=-1, keepdims=True) + 1e-9)

        flat_expert_idx = top_k_idx.reshape(-1)                        # (n_assign,)
        flat_gate = gate.reshape(-1)                                    # (n_assign,)
        flat_token_idx = jnp.repeat(jnp.arange(num_tokens), K)          # (n_assign,)

        mean_probs = jnp.mean(router_probs, axis=0)
        expert_gate_frac = jnp.zeros(E, dtype=flat_gate.dtype).at[flat_expert_idx].add(flat_gate) / num_tokens
        self.sow("losses", "aux_loss", E * jnp.sum(mean_probs * expert_gate_frac))
        self.sow("losses", "z_loss", jnp.mean(jnp.square(jax.scipy.special.logsumexp(router_logits, axis=-1))))
        expert_assign_frac = jnp.zeros(E, dtype=flat_gate.dtype).at[flat_expert_idx].add(1.0) / n_assign
        self.sow("losses", "expert_utilization", expert_assign_frac)

        capacity = max(1, int(self.cfg.moe_capacity_factor * num_tokens * K / E))

        sort_order = jnp.argsort(flat_expert_idx)                       # (n_assign,) -- stable by default
        sorted_expert = flat_expert_idx[sort_order]

        expert_counts = jnp.zeros(E, dtype=jnp.int32).at[flat_expert_idx].add(1)
        expert_start = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(expert_counts)[:-1]])

        pos_in_bucket_sorted = jnp.arange(n_assign) - expert_start[sorted_expert]
        valid_sorted = pos_in_bucket_sorted < capacity

        dest_row = sorted_expert * capacity + jnp.minimum(pos_in_bucket_sorted, capacity - 1)

        gathered_x = flat_x[flat_token_idx][sort_order]                 # (n_assign, d)
        buffer = jnp.zeros((E * capacity, d), dtype=flat_x.dtype)
        buffer = buffer.at[dest_row].add(jnp.where(valid_sorted[:, None], gathered_x, 0.0))
        expert_inputs = buffer.reshape(E, capacity, d)

        run_experts = nn.vmap(
            ExpertPack,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, None),
            out_axes=0,
        )(cfg=self.cfg, name="experts_block")
        expert_outputs = run_experts(expert_inputs, deterministic)      # (E, capacity, d)

        flat_expert_outputs = expert_outputs.reshape(E * capacity, d)
        gathered_out_sorted = flat_expert_outputs[dest_row]             # (n_assign, d)
        gathered_out_sorted = jnp.where(valid_sorted[:, None], gathered_out_sorted, 0.0)

        unsort_order = jnp.argsort(sort_order)                          # inverse permutation
        gathered_out = gathered_out_sorted[unsort_order]                # back to assignment order
        weighted_out = gathered_out * flat_gate[:, None]

        flat_outputs = jnp.zeros_like(flat_x).at[flat_token_idx].add(weighted_out)
        return flat_outputs.reshape(b, l, d)


# ==========================================
# Delta-Attention Residual Block
# ==========================================
class DeltaAttentionResidualBlockJ(nn.Module):
    cfg: ModelConfig
    layer_idx: int

    @nn.compact
    def __call__(self, current_x, history_deltas, causal_mask, cos, sin, deterministic: bool = True, rngs=None):
        norm_1 = nn.RMSNorm(epsilon=1e-6, name="norm_1")(current_x)
        gdn_out = GatedDeltaNet2J(cfg=self.cfg, name="gdn")(norm_1)
        mamba_out = Mamba2J(cfg=self.cfg, name="mamba")(norm_1)
        mla_out = MLAJ(cfg=self.cfg, name="mla")(
            norm_1, causal_mask, cos, sin, 
            deterministic=deterministic, rngs=rngs
        )

        alpha = jax.nn.softmax(self.param("alpha", nn.initializers.zeros, (3,)))
        current_delta = jnp.einsum("i,ibld->bld", alpha, jnp.stack([gdn_out, mamba_out, mla_out], axis=0))

        updated_history = history_deltas.at[self.layer_idx].set(current_delta)

        q_route = nn.Dense(self.cfg.d_latent, use_bias=False, name="q_route")(current_x)
        k_route = nn.Dense(self.cfg.d_latent, use_bias=False, name="k_route")(updated_history)
        routing_scores = jnp.einsum("bld,vbld->blv", q_route, k_route) / jnp.sqrt(self.cfg.d_latent)

        depth_mask = jnp.arange(self.cfg.num_layers) <= self.layer_idx
        routing_scores = jnp.where(depth_mask[None, None, :], routing_scores, -1e9)
        routing_weights = jax.nn.softmax(routing_scores, axis=-1)

        moe_in = current_x + jnp.einsum("blv,vbld->bld", routing_weights, updated_history)
        norm_2 = nn.RMSNorm(epsilon=1e-6, name="norm_2")(moe_in)
        moe_out = MoEJ(cfg=self.cfg, name="moe")(norm_2, deterministic=deterministic, rngs=rngs)
        return moe_in + moe_out, updated_history


# ==========================================
# Block of 2 consecutive Delta-Attention Residual layers, sharing one remat scope
# ==========================================
#
# ЗАЧЕМ: remat на КАЖДЫЙ отдельный слой (было раньше) даёт максимальную экономию
# памяти (пик активаций одного слоя), но каждый remat -- это отдельная граница
# recompute при backward: JAX должен заново затрассировать forward для КАЖДОГО
# из 22 слоёв по отдельности. Объединение по 2 слоя в один remat-блок снижает
# число таких границ вдвое (22 -> 11), что уменьшает накладные расходы на
# retracing/scheduling, ценой того, что теперь ОДИН remat-блок держит в памяти
# активации СРАЗУ ДВУХ слоёв вместо одного (по-прежнему на порядок меньше, чем
# без remat вообще, где держатся активации всех 22 слоёв разом).
#
# ВАЖНО про чекпоинты: это МЕНЯЕТ дерево параметров -- было плоское layer_0,
# layer_1, ..., layer_21; станет layer_pair_0/layer_0, layer_pair_0/layer_1,
# layer_pair_2/layer_2, ... Старые сохранённые Orbax-чекпоинты с плоской
# структурой НЕ загрузятся без ремаппинга путей.
class DeltaResidualBlockPairJ(nn.Module):
    cfg: ModelConfig
    layer_idx_0: int  # first of the pair; the second is layer_idx_0 + 1

    @nn.compact
    def __call__(self, current_x, history_deltas, causal_mask, cos, sin, deterministic: bool = True, rngs=None):
        current_x, history_deltas = DeltaAttentionResidualBlockJ(
            cfg=self.cfg, layer_idx=self.layer_idx_0, name=f"layer_{self.layer_idx_0}"
        )(current_x, history_deltas, causal_mask, cos, sin, deterministic, rngs)
        current_x, history_deltas = DeltaAttentionResidualBlockJ(
            cfg=self.cfg, layer_idx=self.layer_idx_0 + 1, name=f"layer_{self.layer_idx_0 + 1}"
        )(current_x, history_deltas, causal_mask, cos, sin, deterministic, rngs)
        return current_x, history_deltas


# ==========================================
# Full model
# ==========================================
class FullHybridMoEModel(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, input_ids, deterministic: bool = True, rngs=None, mesh=None):
        b, l = input_ids.shape
        embed_layer = nn.Embed(num_embeddings=self.cfg.vocab_size, features=self.cfg.d_model, name="embed")
        x = embed_layer(input_ids)
        causal_mask = jnp.tril(jnp.ones((l, l))).astype(jnp.bool_)[None, None, :, :]

        d_head = self.cfg.d_model // self.cfg.n_heads
        cos, sin = RoPEEmbedding(dim=d_head)(l)

        history_deltas = jnp.zeros((self.cfg.num_layers, b, l, self.cfg.d_model), dtype=x.dtype)

        # static_argnums index 6 = `deterministic` in both wrappers below (same
        # __call__ signature position: self, current_x, history_deltas,
        # causal_mask, cos, sin, deterministic, rngs).
        RematPair = nn.remat(DeltaResidualBlockPairJ, static_argnums=(6,))
        RematSingle = nn.remat(DeltaAttentionResidualBlockJ, static_argnums=(6,))

        num_full_pairs = self.cfg.num_layers // 2
        for p in range(num_full_pairs):
            i = p * 2
            # Добавляем mesh в rngs
            if rngs is not None:
                rngs_with_mesh = dict(rngs)
                rngs_with_mesh["mesh"] = mesh
            else:
                rngs_with_mesh = {"mesh": mesh}
            x, history_deltas = RematPair(
                cfg=self.cfg, layer_idx_0=i, name=f"layer_pair_{i}"
            )(x, history_deltas, causal_mask, cos, sin, deterministic, rngs_with_mesh)

        # odd num_layers: handle the leftover single layer with per-layer remat
        if self.cfg.num_layers % 2 == 1:
            i = num_full_pairs * 2
            rngs_with_mesh = dict(rngs) if rngs is not None else {}
            rngs_with_mesh["mesh"] = mesh
            x, history_deltas = RematSingle(
                cfg=self.cfg, layer_idx=i, name=f"layer_{i}"
            )(x, history_deltas, causal_mask, cos, sin, deterministic, rngs_with_mesh)
        final = nn.RMSNorm(epsilon=1e-6, name="final_norm")(x)
        if self.cfg.tie_embeddings:
            logits = embed_layer.attend(final)
        else:
            logits = nn.Dense(self.cfg.vocab_size, use_bias=False, name="lm_head")(final)
        return logits
