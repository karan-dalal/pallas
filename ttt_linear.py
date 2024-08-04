from functools import partial

import matplotlib.pyplot as plt

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp
import timeit
import pdb

seq_len = 2048
mini_batch_size = 16
head_dim = 64

key = jax.random.PRNGKey(0) 
XQ = jax.random.uniform(key, (seq_len // mini_batch_size, mini_batch_size, head_dim))
XQ_full = XQ.reshape(seq_len, head_dim)
key, subkey = jax.random.split(key)
XK = jax.random.uniform(key, (seq_len // mini_batch_size, mini_batch_size, head_dim))
XK_full = XK.reshape(seq_len, head_dim)
key, subkey = jax.random.split(key)
XV = jax.random.uniform(key, (seq_len // mini_batch_size, mini_batch_size, head_dim))
XV_full = XV.reshape(seq_len, head_dim)
key, subkey = jax.random.split(key)
eta = jax.random.uniform(subkey, (seq_len // mini_batch_size, mini_batch_size, mini_batch_size))
eta_full = eta.reshape(seq_len, mini_batch_size)

W1_init = jax.random.uniform(key, (head_dim, head_dim))
key, subkey = jax.random.split(key)
b1_init = jax.random.uniform(subkey, (1, head_dim))

key, subkey = jax.random.split(key)
ttt_norm_scale = jax.random.uniform(subkey, (head_dim,))
key, subkey = jax.random.split(key)
ttt_norm_bias = jax.random.uniform(subkey, (head_dim,))

def apply_layernorm(scale, bias, input):
    eps = 1e-6
    mean = jnp.mean(input, axis=-1, keepdims=True)
    var = jnp.var(input, axis=-1, keepdims=True)
    normalized_input = (input - mean) / jnp.sqrt(var + eps)
    return scale * normalized_input + bias

@jax.jit
def m1_forward(
    XQ_mini_batch,
    XK_mini_batch,
    XV_mini_batch,
    eta_mini_batch,
    ttt_params_mini_batch_init,
    ttt_norm_params
):
    W1_init, b1_init = ttt_params_mini_batch_init
    ttt_norm_scale, ttt_norm_bias = ttt_norm_params

    square_eta_mini_batch = eta_mini_batch # [16, 16]
    last_eta_in_mini_batch = eta_mini_batch[-1][:, None] # [16, 1]   
    
    X1 = XK_mini_batch
    Z1 = X1 @ W1_init + b1_init # [16, 64]
    ttt_norm_out, ttt_norm_vjp = jax.vjp(lambda z: apply_layernorm(ttt_norm_scale, ttt_norm_bias, z), Z1) # [16, 64]

    ssl_target = XV_mini_batch - XK_mini_batch # [16, 64]
    grad_l_wrt_ttt_norm_out = ttt_norm_out - ssl_target # [16, 64]
    grad_l_wrt_Z1 = ttt_norm_vjp(grad_l_wrt_ttt_norm_out)[0] # [16, 64]   

    X1_bar = XQ_mini_batch
    Attn1 = jnp.tril(X1_bar @ X1.transpose(1, 0)) # [16, 16]
    b1_bar = b1_init - (square_eta_mini_batch * jnp.tril(jnp.ones_like(Attn1))) @ grad_l_wrt_Z1 # [16, 64]
    Z1_bar = X1_bar @ W1_init - (square_eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar # [16, 64]
    ttt_norm_out_bar = apply_layernorm(ttt_norm_scale, ttt_norm_bias, Z1_bar) # [16, 64]

    output_mini_batch = X1_bar + ttt_norm_out_bar # [16, 64]    

    W1_bar_last = W1_init - (last_eta_in_mini_batch * X1).transpose(1, 0) @ grad_l_wrt_Z1
    b1_bar_last = b1_init - jnp.sum(last_eta_in_mini_batch * grad_l_wrt_Z1, axis=0, keepdims=True)

    ttt_params_mini_batch_new = (W1_bar_last, b1_bar_last)

    return (ttt_params_mini_batch_new, output_mini_batch)

def m1_forward_kernel(
    XQ_mini_batch_ref,
    XK_mini_batch_ref,
    XV_mini_batch_ref,
    eta_mini_batch_ref,
    W1_init_ref,
    b1_init_ref,
    ttt_norm_scale_ref, 
    ttt_norm_bias_ref,
    o_ref,
):
    W1_init, b1_init = W1_init_ref[...], b1_init_ref[...]
    ttt_norm_scale, ttt_norm_bias = ttt_norm_scale_ref[...], ttt_norm_bias_ref[...]

    X1 = XK_mini_batch_ref[...]
    Z1 = X1 @ W1_init + b1_init # [16, 64]
    ttt_norm_out, ttt_norm_vjp = jax.vjp(lambda z: apply_layernorm(ttt_norm_scale, ttt_norm_bias, z), Z1) # [16, 64]

    ssl_target = XV_mini_batch_ref[...] - X1 # [16, 64]
    grad_l_wrt_ttt_norm_out = ttt_norm_out - ssl_target # [16, 64]
    grad_l_wrt_Z1 = ttt_norm_vjp(grad_l_wrt_ttt_norm_out)[0] # [16, 64]

    eta_mini_batch = eta_mini_batch_ref[...]
    eta_mini_batch_last = eta_mini_batch_ref[-1][:, None]
    
    X1_bar = XQ_mini_batch_ref[...]
    Attn1 = jnp.tril(X1_bar @ X1.transpose(1, 0)) # [16, 16]
    b1_bar = b1_init - (eta_mini_batch * jnp.tril(jnp.ones_like(Attn1))) @ grad_l_wrt_Z1 # [16, 64]
    Z1_bar = X1_bar @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar # [16, 64]
    ttt_norm_out_bar = apply_layernorm(ttt_norm_scale, ttt_norm_bias, Z1_bar) # [16, 64]
    output_mini_batch = X1_bar + ttt_norm_out_bar # [16, 64]

    W1_bar_last = W1_init - (eta_mini_batch_last * X1).transpose(1, 0) @ grad_l_wrt_Z1
    b1_bar_last = b1_init - jnp.sum(eta_mini_batch_last * grad_l_wrt_Z1, axis=0, keepdims=True)

    pl.store(o_ref, (pl.dslice(0, 16), slice(None)), output_mini_batch)
    pl.store(o_ref, (pl.dslice(16, 64), slice(None)), W1_bar_last)
    pl.store(o_ref, (pl.dslice(80, 1), slice(None)), b1_bar_last)

@jax.jit
def m1_pallas(
    XQ_mini_batch,
    XK_mini_batch,
    XV_mini_batch,
    eta_mini_batch,
    ttt_params_mini_batch_init,
    ttt_norm_params,
):
    W1_init, b1_init = ttt_params_mini_batch_init
    ttt_norm_scale, ttt_norm_bias = ttt_norm_params

    pallas_output = pl.pallas_call(
        m1_forward_kernel,
        out_shape=jax.ShapeDtypeStruct((81, 64), XQ_mini_batch.dtype)
    )(
        XQ_mini_batch,
        XK_mini_batch,
        XV_mini_batch,
        eta_mini_batch,
        W1_init,
        b1_init,
        ttt_norm_scale,
        ttt_norm_bias
    )

    W1_bar_last = pallas_output[16:80]
    b1_bar_last = pallas_output[80:81]

    output = pallas_output[:16]
    ttt_params_mini_batch_new = (W1_bar_last, b1_bar_last)

    return (ttt_params_mini_batch_new, output)

def m1_forward_loop_kernel(
    XQ_ref,
    XK_ref,
    XV_ref,
    eta_ref,
    W1_init_ref,
    b1_init_ref,
    ttt_norm_scale_ref, 
    ttt_norm_bias_ref,
    o_ref,
):
    W1_init, b1_init = W1_init_ref[...], b1_init_ref[...]
    ttt_norm_scale, ttt_norm_bias = ttt_norm_scale_ref[...], ttt_norm_bias_ref[...]

    def process_mini_batch(i, curr):
        W1_init, b1_init = curr

        X1 = pl.load(XK_ref, (pl.dslice(i * 16, 16), slice(None)))
        Z1 = X1 @ W1_init + b1_init # [16, 64]
        ttt_norm_out, ttt_norm_vjp = jax.vjp(lambda z: apply_layernorm(ttt_norm_scale, ttt_norm_bias, z), Z1) # [16, 64]

        ssl_target = pl.load(XV_ref, (pl.dslice(i * 16, 16), slice(None))) - X1 # [16, 64]
        grad_l_wrt_ttt_norm_out = ttt_norm_out - ssl_target # [16, 64]
        grad_l_wrt_Z1 = ttt_norm_vjp(grad_l_wrt_ttt_norm_out)[0] # [16, 64]

        eta_mini_batch = pl.load(eta_ref, (pl.dslice(i * 16, 16), slice(None)))
        eta_mini_batch_last = pl.load(eta_ref, (pl.dslice(i * 16, 16), pl.dslice(i * 16 + 15, 1)))
        
        X1_bar = pl.load(XQ_ref, (pl.dslice(i * 16, 16), slice(None)))
        Attn1 = jnp.tril(X1_bar @ X1.transpose(1, 0)) # [16, 16]
        b1_bar = b1_init - (eta_mini_batch * jnp.tril(jnp.ones_like(Attn1))) @ grad_l_wrt_Z1 # [16, 64]
        Z1_bar = X1_bar @ W1_init - (eta_mini_batch * Attn1) @ grad_l_wrt_Z1 + b1_bar # [16, 64]
        ttt_norm_out_bar = apply_layernorm(ttt_norm_scale, ttt_norm_bias, Z1_bar) # [16, 64]
        output_mini_batch = X1_bar + ttt_norm_out_bar # [16, 64]

        pl.store(o_ref, (pl.dslice(i * 16, 16), slice(None)), output_mini_batch)

        W1_bar_last = W1_init - (eta_mini_batch_last * X1).transpose(1, 0) @ grad_l_wrt_Z1
        b1_bar_last = b1_init - jnp.sum(eta_mini_batch_last * grad_l_wrt_Z1, axis=0, keepdims=True)
        return W1_bar_last, b1_bar_last
    
    (W1_bar_last, b1_bar_last) = jax.lax.fori_loop(0, 128, process_mini_batch, (W1_init, b1_init))

    pl.store(o_ref, (pl.dslice(2048, 64), slice(None)), W1_bar_last)
    pl.store(o_ref, (pl.dslice(2112, 1), slice(None)), b1_bar_last)

@jax.jit
def m1_pallas_loop(
    XQ,
    XK,
    XV,
    eta,
    ttt_params_mini_batch_init,
    ttt_norm_params,
):
    W1_init, b1_init = ttt_params_mini_batch_init
    ttt_norm_scale, ttt_norm_bias = ttt_norm_params

    pallas_output = pl.pallas_call(
        m1_forward_loop_kernel,
        out_shape=jax.ShapeDtypeStruct((2113, 64), XQ.dtype)
    )(
        XQ,
        XK,
        XV,
        eta,
        W1_init,
        b1_init,
        ttt_norm_scale,
        ttt_norm_bias
    )

    W1_bar_last = pallas_output[2048:2112]
    b1_bar_last = pallas_output[2112:2113]

    output = pallas_output[:2048]
    ttt_params_mini_batch_new = (W1_bar_last, b1_bar_last)

    return (ttt_params_mini_batch_new, output)

@jax.jit
def run_comparison():
    
    def compute_mini_batch_forward(ttt_params_mini_batch_init, inputs):
        XQ_mini_batch = inputs["XQ"]
        XK_mini_batch = inputs["XK"]
        XV_mini_batch = inputs["XV"]
        eta_mini_batch = inputs["eta"]

        ttt_norm_params = (ttt_norm_scale, ttt_norm_bias)

        ttt_params_last_in_mini_batch, outputs = m1_forward(
            XQ_mini_batch, XK_mini_batch, XV_mini_batch, eta_mini_batch, ttt_params_mini_batch_init, ttt_norm_params
        )

        return ttt_params_last_in_mini_batch, outputs

    def compute_mini_batch_pallas(ttt_params_mini_batch_init, inputs):
        XQ_mini_batch = inputs["XQ"]
        XK_mini_batch = inputs["XK"]
        XV_mini_batch = inputs["XV"]
        eta_mini_batch = inputs["eta"]

        ttt_norm_params = (ttt_norm_scale, ttt_norm_bias)

        ttt_params_last_in_mini_batch, outputs = m1_pallas(
            XQ_mini_batch, XK_mini_batch, XV_mini_batch, eta_mini_batch, ttt_params_mini_batch_init, ttt_norm_params
        )

        return ttt_params_last_in_mini_batch, outputs

    inputs = {"XQ": XQ, "XK": XK, "XV": XV, "eta": eta}
    ttt_params_init = (W1_init, b1_init)

    # Warmup
    _ = jax.block_until_ready(jax.lax.scan(compute_mini_batch_forward, ttt_params_init, inputs))
    _ = jax.block_until_ready(jax.lax.scan(compute_mini_batch_pallas, ttt_params_init, inputs))
    _ = jax.block_until_ready(m1_pallas_loop(XQ_full, XK_full, XV_full, eta_full, (W1_init, b1_init), (ttt_norm_scale, ttt_norm_bias)))

    forward_time = timeit.timeit(lambda: jax.block_until_ready(jax.lax.scan(compute_mini_batch_forward, ttt_params_init, inputs)), number=10) / 10
    pallas_time = timeit.timeit(lambda: jax.block_until_ready(jax.lax.scan(compute_mini_batch_pallas, ttt_params_init, inputs)), number=10) / 10
    pallas_loop_time = timeit.timeit(lambda: jax.block_until_ready(m1_pallas_loop(XQ_full, XK_full, XV_full, eta_full, (W1_init, b1_init), (ttt_norm_scale, ttt_norm_bias))), number=10) / 10

    print(f"Average time for m1_forward: {forward_time:.6f} seconds")
    print(f"Average time for m1_pallas: {pallas_time:.6f} seconds")
    print(f"Average time for m1_pallas_loop: {pallas_loop_time:.6f} seconds")


if __name__ == "__main__":
    run_comparison()