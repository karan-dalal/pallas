import jax
import torch
import jax.numpy as jnp
import numpy as np

from functools import partial
from timeit import timeit
from jax.experimental import pallas as pl

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide CUBLAS support error.

def timing(f, *args):
    f_time = lambda: f(*args)
    timeit(f_time, number=20)
    return timeit(f_time, number=100)

embd_dim = 768
head_dim = 64
heads = 12

key = jax.random.PRNGKey(0)

A = jax.random.normal(key, shape=(heads, head_dim, embd_dim))
A_tranpose = jnp.transpose(A, (0, 2, 1))

B = jax.random.normal(key, shape=(heads, embd_dim, head_dim))
C = jax.random.normal(key, shape=(heads, head_dim, embd_dim))
D = jax.random.normal(key, shape=(heads,embd_dim, head_dim))
W0 = jax.random.normal(key, shape=(heads, head_dim, head_dim))

res_dict = {
    'sequence_length': [],
    'scan': [],
    'kernel_16': [],
    'kernel_hadamard': []
}

sequences = [2**i for i in range(5, 12, 1)]

for seq_len in sequences:
    
    sequence = jax.random.normal(key, shape=(heads, seq_len, embd_dim))
    kernel_output = jax.random.normal(key, shape=(seq_len, head_dim))

    @jax.jit
    @jax.vmap
    def scan_forward(sequence, A, B, C, D, W0):

      def inner_forward(token, W):
        token_transformed = token @ B @ W @ A
        return 0.5 * ((token_transformed - token) ** 2).sum()

      def new_forward(token, W):
        return token @ D @ W @ C

      inner_grad = jax.grad(inner_forward, argnums=1)

      def body(carry, token):
        W = carry

        grad = inner_grad(token, W)
        W_new = W - grad
        token_transformed = new_forward(token, W_new)

        return W_new, token_transformed

      _, output = jax.lax.scan(body, W0, sequence)
      return output

    def kernel_16(seq_ref, seq_b_ref, seq_d_ref, A_ref, W_ref, o_ref):
      sequence_length, embd_dim = seq_ref.shape
      head_dim = W_ref.shape[0]
      BLOCK_SIZE = 256

      W = W_ref[:]

      def body(i, W):
        token_b = pl.load(seq_b_ref, (pl.dslice(i, 1), slice(None)))
        token_d = pl.load(seq_d_ref, (pl.dslice(i, 1), slice(None)))

        token_b = jnp.repeat(token_b, 16, axis=0)
        token_d = jnp.repeat(token_d, 16, axis=0)
        
        token_pre_A = token_b @ W
        
        def tile_gradient(j, accum):
          A_col = pl.load(A_ref, (slice(None), pl.dslice(j * BLOCK_SIZE, BLOCK_SIZE)))
          
          token_j = pl.load(seq_ref, (pl.dslice(i, 1), pl.dslice(j * BLOCK_SIZE, BLOCK_SIZE)))
          token_j = jnp.repeat(token_j, 16, axis=0)

          accum += (token_pre_A @ A_col - token_j) @ A_col.T
          return accum

        accum_grad = jax.lax.fori_loop(0, embd_dim // BLOCK_SIZE, tile_gradient, jnp.zeros((16, head_dim)))
        gradient = (token_b.T @ accum_grad) / 16
        W_new = W - gradient
        token_pre_C = token_d @ W_new
        pl.store(o_ref, (pl.dslice(i, 16), slice(None)), token_pre_C)

        return W_new

      W = jax.lax.fori_loop(0, sequence_length, body, W)

    def kernel_hadamard(seq_ref, seq_b_ref, seq_d_ref, A_ref, W_ref, o_ref):
      sequence_length, embd_dim = seq_ref.shape
      head_dim = W_ref.shape[0]
      BLOCK_SIZE = 512 # NOTE: Will not give correct result, not multiple of embedding dimension.

      W = W_ref[:]

      def body(i, W):
        token_b = pl.load(seq_b_ref, (pl.dslice(i, 1), slice(None)))
        token_d = pl.load(seq_d_ref, (pl.dslice(i, 1), slice(None)))

        token_pre_A = jnp.sum(token_b.T * W, axis=0, keepdims=True)

        def tile_gradient(j, accum):
          # A_col = pl.load(A_ref, (slice(None), pl.dslice(j * BLOCK_SIZE, BLOCK_SIZE)))
          A_tranpose_row = pl.load(A_ref, (pl.dslice(j * BLOCK_SIZE, BLOCK_SIZE), slice(None)))
          token = pl.load(seq_ref, (pl.dslice(i, 1), pl.dslice(j * BLOCK_SIZE, BLOCK_SIZE)))

          loss = jnp.sum(token_pre_A.T * A_tranpose_row.T, axis=0, keepdims=True) - token
          accum += jnp.sum(loss.T * A_tranpose_row, axis=0, keepdims=True)
          return accum

        accum_grad = jax.lax.fori_loop(0, embd_dim // BLOCK_SIZE, tile_gradient, jnp.zeros((1, head_dim)))
        gradient = token_b.T * accum_grad

        W_new = W - gradient
        token_pre_C = jnp.sum(token_d.T * W_new, axis=0, keepdims=True)
        pl.store(o_ref, (pl.dslice(i, 1), slice(None)), token_pre_C)

        return W_new

      W = jax.lax.fori_loop(0, sequence_length, body, W)

    @jax.jit
    def kernel_16_forward(sequence, A, B, C, D, W0):
      # Pre-compute "noise" transformations
      sequence_B = sequence @ B
      sequence_D = sequence @ D

      # Initialize and call kernel
      kernel_forward = jax.vmap(pl.pallas_call(kernel_16, out_shape=jax.ShapeDtypeStruct(kernel_output.shape, kernel_output.dtype)))
      output = kernel_forward(sequence, sequence_B, sequence_D, A, W0)
      return output @ C

    @jax.jit
    def kernel_hadamard_forward(sequence, A, B, C, D, W0):
      # Pre-compute "noise" transformations
      sequence_B = sequence @ B
      sequence_D = sequence @ D

      # Initialize and call kernel
      kernel_forward = jax.vmap(pl.pallas_call(kernel_hadamard, out_shape=jax.ShapeDtypeStruct(kernel_output.shape, kernel_output.dtype)))
      output = kernel_forward(sequence, sequence_B, sequence_D, A, W0)

      return output @ C

    res_dict['sequence_length'].append(seq_len)
    res_dict['scan'].append(timing(scan_forward, sequence, A, B, C, D, W0) * 1000)
    res_dict['kernel_16'].append(timing(kernel_16_forward, sequence, A, B, C, D, W0) * 1000)
    res_dict['kernel_hadamard'].append(timing(kernel_hadamard_forward, sequence, A_tranpose, B, C, D, W0) * 1000)

print(res_dict)
torch.save(res_dict, 'results/m1_gradient_kernel_forward_contiguous.pth') 