import jax
import torch
import jax.numpy as jnp
import numpy as np

from jax.experimental import pallas as pl

key = jax.random.PRNGKey(0)

matrix = jax.random.normal(key, shape=(64, 768))
batch = jax.random.normal(key, shape=(16, 64))
output = jnp.zeros([16, 64])


def kernel(batch_ref, matrix_ref, o_ref):
    batch = batch_ref[:]
    BLOCK_SIZE = 64
    
    def accum(i, accumulation):
        matrix_col = pl.load(matrix_ref, (slice(None), pl.dslice(i * BLOCK_SIZE, BLOCK_SIZE)))
        accumulation += batch @ matrix_col 
        return accumulation

    output = jax.lax.fori_loop(0, matrix_ref.shape[1] // BLOCK_SIZE, accum, jnp.zeros((16, BLOCK_SIZE)))
    o_ref[:] = output

launch_kernel = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(output.shape, output.dtype))

def baseline(batch, matrix):
    BLOCK_SIZE = 64

    def accum(i, accumulation):
        matrix_col = jax.lax.dynamic_slice(matrix, (0, i * BLOCK_SIZE), (matrix.shape[0], 64))
        accumulation += batch @ matrix_col
        return accumulation

    output = jax.lax.fori_loop(0, matrix.shape[1] // BLOCK_SIZE, accum, jnp.zeros((16, BLOCK_SIZE)))
    return output

output1 = baseline(batch, matrix)
output2 = launch_kernel(batch, matrix)

print("Outside Kernel")
print(output1)

print("Inside Kernel")
print(output2)