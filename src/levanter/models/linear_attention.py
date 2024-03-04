import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

import haliax as hax
from haliax import Axis, AxisSelection, AxisSelector, NamedArray

TILE_DIM = 128
EPS = 1e-6
INTERPRET = False
DEBUG = False

@jax.custom_vjp
def attn(q: jax.Array, k: jax.Array, v: jax.Array, kv_carry: jax.Array, k_carry: jax.Array):
    return pl.pallas_call(
        attn_fwd_kernel,
        interpret=INTERPRET,
        debug=DEBUG,
        out_shape=[jax.ShapeDtypeStruct(v.shape, v.dtype), jax.ShapeDtypeStruct(kv_carry.shape, kv_carry.dtype), jax.ShapeDtypeStruct(k_carry.shape, k_carry.dtype)],
        grid=(q.shape[0], q.shape[1], q.shape[-2] // TILE_DIM),
        in_specs=[
            pl.BlockSpec(lambda b, h, s: (b, h, s, 0), (1, 1, TILE_DIM, q.shape[-1])),
            pl.BlockSpec(lambda b, h, s: (b, h, s, 0), (1, 1, TILE_DIM, k.shape[-1])),
            pl.BlockSpec(lambda b, h, s: (b, h, s, 0), (1, 1, TILE_DIM, v.shape[-1])),
            pl.BlockSpec(lambda b, h, _: (b, h, 0, 0), (1, 1, k.shape[-1], v.shape[-1])),
            pl.BlockSpec(lambda b, h, _: (b, h, 0, 0), (1, 1, k.shape[-1], k.shape[-1])),
        ],
        out_specs=[
            pl.BlockSpec(lambda b, h, s: (b, h, s, 0), (1, 1, TILE_DIM, v.shape[-1])),
            pl.BlockSpec(lambda b, h, _: (b, h, 0, 0), (1, 1, k.shape[-1], v.shape[-1])),
            pl.BlockSpec(lambda b, h, _: (b, h, 0, 0), (1, 1, k.shape[-1], k.shape[-1])),
        ],
        mosaic_params=dict(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(q, k, v, kv_carry, k_carry)

def attn_bwd_pallas(q: jax.Array, k: jax.Array, v: jax.Array, kv_carry: jax.Array, k_carry: jax.Array, dy: jax.Array, dkv_carry: jax.Array, dk_carry: jax.Array):
    return pl.pallas_call(
        attn_bwd_kernel,
        interpret=INTERPRET,
        debug=DEBUG,
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype), 
            jax.ShapeDtypeStruct(k.shape, k.dtype), 
            jax.ShapeDtypeStruct(v.shape, v.dtype), 
        ],
        grid=(q.shape[0], q.shape[1], q.shape[-2] // TILE_DIM),
        in_specs=[
            pl.BlockSpec(lambda b, h, s: (b, h, q.shape[-2] // TILE_DIM - s - 1, 0), (1, 1, TILE_DIM, q.shape[-1])),
            pl.BlockSpec(lambda b, h, s: (b, h, q.shape[-2] // TILE_DIM - s - 1, 0), (1, 1, TILE_DIM, k.shape[-1])),
            pl.BlockSpec(lambda b, h, s: (b, h, q.shape[-2] // TILE_DIM - s - 1, 0), (1, 1, TILE_DIM, v.shape[-1])),
            pl.BlockSpec(lambda b, h, _: (b, h, 0, 0), (1, 1, k.shape[-1], v.shape[-1])),
            pl.BlockSpec(lambda b, h, _: (b, h, 0, 0), (1, 1, k.shape[-1], k.shape[-1])),
            pl.BlockSpec(lambda b, h, s: (b, h, q.shape[-2] // TILE_DIM - s - 1, 0), (1, 1, TILE_DIM, v.shape[-1])),
            pl.BlockSpec(lambda b, h, _: (b, h, 0, 0), (1, 1, k.shape[-1], v.shape[-1])),
            pl.BlockSpec(lambda b, h, _: (b, h, 0, 0), (1, 1, k.shape[-1], k.shape[-1])),
        ],
        out_specs=[
            pl.BlockSpec(lambda b, h, s: (b, h, q.shape[-2] // TILE_DIM - s - 1, 0), (1, 1, TILE_DIM, q.shape[-1])),
            pl.BlockSpec(lambda b, h, s: (b, h, q.shape[-2] // TILE_DIM - s - 1, 0), (1, 1, TILE_DIM, k.shape[-1])),
            pl.BlockSpec(lambda b, h, s: (b, h, q.shape[-2] // TILE_DIM - s - 1, 0), (1, 1, TILE_DIM, v.shape[-1])),
        ],
        mosaic_params=dict(dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(q, k, v, kv_carry, k_carry, dy, dkv_carry, dk_carry)

def attn_fwd_kernel(q_ref, k_ref, v_ref, kv_carry_in_ref, k_carry_in_ref, y_ref, kv_carry_out_ref, k_carry_out_ref):
    # Load inputs
    operands_dtype = q_ref.dtype
    q = q_ref[0, 0, :, :].astype(jnp.float32)
    k = k_ref[0, 0, :, :].astype(jnp.float32)
    v = v_ref[0, 0, :, :].astype(jnp.float32)
    kv_carry = kv_carry_in_ref[0, 0, :, :].astype(jnp.float32)
    k_carry = k_carry_in_ref[0, 0, :, :].astype(jnp.float32)

    attn_scores = jnp.tril(q @ k.T)
    ones = jnp.ones_like(kv_carry)
    y = (attn_scores @ v + q @ kv_carry) / (attn_scores @ ones + q @ k_carry + EPS)
    kv_carry += k.T @ v
    k_carry += k.T @ ones

    # Store outputs
    y_ref[0, 0, :, :] = y.astype(operands_dtype)
    kv_carry_in_ref[0, 0, :, :] = kv_carry.astype(operands_dtype)
    k_carry_in_ref[0, 0, :, :] = k_carry.astype(operands_dtype)
    kv_carry_out_ref[0, 0, :, :] = kv_carry.astype(operands_dtype)
    k_carry_out_ref[0, 0, :, :] = k_carry.astype(operands_dtype)

def attn_bwd_kernel(q_ref, k_ref, v_ref, kv_carry_ref, k_carry_ref, dy_ref, dkv_carry_ref, dk_carry_ref,
                    dq_ref, dk_ref, dv_ref):
    # Load inputs
    operands_dtype = q_ref.dtype
    q = q_ref[0, 0, :, :].astype(jnp.float32)
    k = k_ref[0, 0, :, :].astype(jnp.float32)
    v = v_ref[0, 0, :, :].astype(jnp.float32)
    kv_carry = kv_carry_ref[0, 0, :, :].astype(jnp.float32)
    k_carry = k_carry_ref[0, 0, :, :].astype(jnp.float32)
    dy = dy_ref[0, 0, :, :].astype(jnp.float32)
    dkv_carry = dkv_carry_ref[0, 0, :, :].astype(jnp.float32)
    dk_carry = dk_carry_ref[0, 0, :, :].astype(jnp.float32)

    ones = jnp.ones_like(kv_carry)

    kv_carry -= k.T @ v
    k_carry -= k.T @ ones

    attn_scores = jnp.tril(q @ k.T)
    numerator = attn_scores @ v + q @ kv_carry
    denominator = attn_scores @ ones + q @ k_carry + EPS

    # Given dy
    # Calculate gradients for denominator and numerator first
    d_numerator = dy / (denominator)
    d_denominator = -dy * (numerator) / (denominator**2)

    # Calculate gradients for attn_scores, v, q, kv_carry, k_carry
    d_attn_scores_numerator = d_numerator @ v.T
    d_attn_scores_denominator = d_denominator @ ones
    d_attn_scores = jnp.tril(d_attn_scores_numerator + d_attn_scores_denominator)

    dv_numerator = attn_scores.T @ d_numerator
    dv_dkv_carry = k @ dkv_carry
    dv = dv_numerator + dv_dkv_carry

    dq_d_attn_scores = d_attn_scores @ k
    dq_kv_carry = d_numerator @ kv_carry.T
    dq_k_carry = d_denominator @ k_carry.T
    dq = dq_d_attn_scores + dq_kv_carry + dq_k_carry

    dk_d_attn_scores = d_attn_scores.T @ q
    dk_dkv_carry = v @ dkv_carry.T
    dk_dk_carry = ones @ dk_carry.T
    dk = dk_d_attn_scores + dk_dkv_carry + dk_dk_carry

    dkv_carry += q.T @ d_numerator
    dk_carry += q.T @ d_denominator

    # Store outputs
    dv_ref[0, 0, :, :] = dv.astype(operands_dtype)
    dq_ref[0, 0, :, :] = dq.astype(operands_dtype)
    dkv_carry_ref[0, 0, :, :] = dkv_carry.astype(operands_dtype)
    dk_carry_ref[0, 0, :, :] = dk_carry.astype(operands_dtype)
    dk_ref[0, 0, :, :] = dk.astype(operands_dtype)
    kv_carry_ref[0, 0, :, :] = kv_carry.astype(operands_dtype)
    k_carry_ref[0, 0, :, :] = k_carry.astype(operands_dtype)

def attn_fwd(q: jax.Array, k: jax.Array, v: jax.Array, kv_carry: jax.Array, k_carry: jax.Array):
    y, kv_carry_new, k_carry_new = attn(q, k, v, kv_carry, k_carry)
    return (y, kv_carry_new, k_carry_new), (q, k, v, kv_carry_new, k_carry_new)

def attn_bwd(res, grad):
    q, k, v, kv_carry, k_carry = res
    dy, _, _ = grad
    dkv_carry = jnp.zeros_like(kv_carry)
    dk_carry = jnp.zeros_like(k_carry)
    dq, dk, dv = attn_bwd_pallas(q, k, v, kv_carry, k_carry, dy, dkv_carry, dk_carry)
    return dq, dk, dv, jnp.zeros_like(kv_carry), jnp.zeros_like(k_carry)

attn.defvjp(attn_fwd, attn_bwd)

def linear_attention(
    query: NamedArray,
    key: NamedArray,
    value: NamedArray,
) -> NamedArray:
    q = query.array
    k = key.array
    v = value.array
    kv_carry = jnp.zeros_like(k)
    k_carry = jnp.zeros_like(k)
    y, _, _ = attn(q, k, v, kv_carry, k_carry)
    named_y = hax.named(y, value.axes)
    return named_y
