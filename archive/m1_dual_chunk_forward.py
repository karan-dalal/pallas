import jax
import torch
import jax.numpy as jnp
import numpy as np

from functools import partial
from timeit import timeit
from jax.experimental import pallas as pl

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hide CUBLAS support error.

embd_dim = 768
head_dim = 64
heads = 12

key = jax.random.PRNGKey(0)
key1, key2 = jax.random.split(key, 2)

A = jax.random.normal(key1, shape=(heads, head_dim, embd_dim))
A_T = jax.random.normal(key1, shape=(heads, embd_dim, head_dim))

B = jax.random.normal(key1, shape=(heads, embd_dim, head_dim))
C = jax.random.normal(key2, shape=(heads, embd_dim, head_dim))
O = jax.random.normal(key2, shape=(heads, head_dim, embd_dim))
W0 = jax.random.normal(key, shape=(heads, head_dim, head_dim))

def timing(f, *args):
    f_time = lambda: f(*args)
    timeit(f_time, number=20)
    return timeit(f_time, number=100)

chunk_dict = {
    'chunk_size': [],
    'results': [],
}

# chunks = [2**i for i in range(2, 10)]
# chunks.insert(0, 1)
chunks = [16]
sequences = [1024]

for chunk_size in chunks:

    res_dict = {
        'sequence_length': [],
        'baseline': [],
        'dual': [],
        'dual_kernel': [],
    }

    for seq_len in sequences:

        sequence = jax.random.normal(key, shape=(heads, seq_len, embd_dim))
        kernel_output = jnp.zeros([seq_len, head_dim])

        @jax.jit
        @jax.vmap
        def baseline(sequence, A, B, C, O, W0):
            
            def inner_gradient(W, token):
                token_transformed = token @ B @ W @ A
                return 0.5 * ((token_transformed - token) ** 2).sum()

            inner_grad = jax.grad(inner_gradient, argnums=0)

            def body(carry, token_chunk):
                W = carry
                    
                grad_chunk = jax.vmap(partial(inner_grad, W))(token_chunk)
                token_chunk = token_chunk @ C

                def inner_forward(carry, input):
                    W = carry
                    token, gradient = input

                    W_new = W - gradient
                    output = token @ W_new
                    return W_new, output

                W_final, token_transformed = jax.lax.scan(inner_forward, W, (token_chunk, grad_chunk))
                token_transformed = token_transformed @ O
                
                return W_final, token_transformed

            sequence = jnp.reshape(sequence, (seq_len // chunk_size, chunk_size, embd_dim))
            W_final, output = jax.lax.scan(body, W0, sequence)
            output = jnp.reshape(output, (seq_len, embd_dim))
            return output

        @jax.jit
        def dual(sequence, A, B, C, O, W0):
            
            sequenceChunked = jnp.reshape(sequence, (heads, seq_len // chunk_size, chunk_size, embd_dim))
            sequenceB = jnp.reshape(sequence @ B, (heads, seq_len // chunk_size, chunk_size, head_dim))
            sequenceC = jnp.reshape(sequence @ C, (heads, seq_len // chunk_size, chunk_size, head_dim))

            @jax.vmap
            def parallelize_over_heads(sequenceChunked, sequenceB, sequenceC, W0, A):
                
                def body(carry, inputs):
                    sequence_chunk, sequenceB_chunk, sequenceC_chunk = inputs
                    W = carry

                    Attn = jnp.tril(sequenceC_chunk @ sequenceB_chunk.T) 
                    P = (sequenceB_chunk @ W @ A - sequence_chunk) @ A.T
                    sequenceCW_chunk = sequenceC_chunk @ W - Attn @ P

                    W_new = W - sequenceB_chunk.T @ P
                    return W_new, sequenceCW_chunk

                W_final, sequenceCW = jax.lax.scan(body, W0, (sequenceChunked, sequenceB, sequenceC))
                return sequenceCW
            
            sequenceCW = parallelize_over_heads(sequenceChunked, sequenceB, sequenceC, W0, A)
            head_output = jnp.reshape(sequenceCW, (heads, seq_len, head_dim)) @ O
            return head_output            


        def kernel(seq_ref, seq_b_ref, seq_c_ref, A_T_ref, W_ref, o_ref):
            seq_len, embd_dim = seq_ref.shape
            head_dim = W_ref.shape[0]
            W = W_ref[:]

            CHUNK_SIZE = chunk_size
            BLOCK_SIZE = 256

            def body(i, W):
                chunk_b = pl.load(seq_b_ref, (pl.dslice(i * CHUNK_SIZE, CHUNK_SIZE), slice(None)))
                chunk_c = pl.load(seq_c_ref, (pl.dslice(i * CHUNK_SIZE, CHUNK_SIZE), slice(None)))

                chunk_bw = chunk_b @ W

                def tile_P(j, P):
                    A_T_row = pl.load(A_T_ref, (pl.dslice(j * BLOCK_SIZE, BLOCK_SIZE), slice(None)))
                    chunk = pl.load(seq_ref, (pl.dslice(i * CHUNK_SIZE, CHUNK_SIZE), pl.dslice(j * BLOCK_SIZE, BLOCK_SIZE)))
                    P += (chunk_bw @ A_T_row.T - chunk) @ A_T_row
                    return P

                P = jax.lax.fori_loop(0, embd_dim // BLOCK_SIZE, tile_P, jnp.zeros((CHUNK_SIZE, head_dim))) # NOTE: Pallas has an annoying bug here.

                tril = jnp.arange(CHUNK_SIZE)
                attn = jnp.where(tril[None, :] <= tril[:, None], chunk_c @ chunk_b.T, 0)
                pl.store(o_ref, (pl.dslice(i * CHUNK_SIZE, CHUNK_SIZE), slice(None)), chunk_c @ W - attn @ P)
                
                W -= chunk_b.T @ P
                return W

            W = jax.lax.fori_loop(0, seq_len // CHUNK_SIZE, body, W)

        launch_kernel = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(kernel_output.shape, kernel_output.dtype))

        @jax.jit
        def dual_kernel(sequence, A_T, B, C, O, W0):
            sequenceB = sequence @ B
            sequenceC = sequence @ C

            @jax.vmap
            def parallelize_over_heads(sequence, sequenceB, sequenceC, W0, A_T):                
                return launch_kernel(sequence, sequenceB, sequenceC, A_T, W0)
            
            sequenceCW = parallelize_over_heads(sequence, sequenceB, sequenceC, W0, A_T)

            head_output = jnp.reshape(sequenceCW, (heads, seq_len, head_dim)) @ O
            return head_output

        res_dict['sequence_length'].append(seq_len)
        res_dict['baseline'].append(timing(baseline, sequence, A, B, C, O, W0) * 1000)
        res_dict['dual'].append(timing(dual, sequence, A, B, C, O, W0) * 1000)
        res_dict['dual_kernel'].append(timing(dual_kernel, sequence, A_T, B, C, O, W0) * 1000)

    chunk_dict['chunk_size'].append(chunk_size)
    chunk_dict['results'].append(res_dict)

print(chunk_dict)
