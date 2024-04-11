import matplotlib.pyplot as plt
import torch

def plot_sequence_x():
    file_path = '/nlp/scr/yusun/data/karan/ttt-gpt/pallas/results/m1_cumsum_forward_verified.pth'
    data = torch.load(file_path)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i, chunk_size in enumerate(data['chunk_size']):
        results = data['results'][i]

        axs[i // 2, i % 2].plot(results['sequence_length'], results['scan'], label=f'Scan')
        axs[i // 2, i % 2].plot(results['sequence_length'], results['kernel_hadamard'], label=f'Hadamard Kernel')
        axs[i // 2, i % 2].plot(results['sequence_length'], results['kernel_16'], label=f'Tensor Kernel')

        axs[i // 2, i % 2].set_xlabel('Sequence Length')
        axs[i // 2, i % 2].set_ylabel('Time (ms)')
        axs[i // 2, i % 2].set_title(f'Chunk Size = {chunk_size}')
        axs[i // 2, i % 2].legend()

    fig.suptitle('M1 Fused CumSum Kernel – 12 Heads (Full Dimension)')
    plt.tight_layout()
    plt.savefig('/nlp/scr/yusun/data/karan/ttt-gpt/pallas/results/m1_fused_cumsum_forward.png', dpi=300)

def plot_chunks():
    file_path = '/nlp/scr/yusun/data/karan/ttt-gpt/pallas/results/m1_fused_kernel_forward.pth'
    data = torch.load(file_path)

    block_file_path = '/nlp/scr/yusun/data/karan/ttt-gpt/pallas/results/m1_fused_kernel_forward_blocks.pth'
    block_data = torch.load(block_file_path)

    chunk_sizes = []

    fused_scan = []
    fused_kernel_hadamard = []
    fused_kernel_16 = []

    cumsum = []
    cumsum_kernel = []
    block_cumsum_kernel = []

    for i, chunk_size in enumerate(data['chunk_size']):
        chunk_sizes.append(chunk_size)
        results = data['results'][i]
        block_results = block_data['results'][i]

        fused_scan.append(results['fused_scan'])
        fused_kernel_hadamard.append(results['fused_kernel_hadamard'])
        fused_kernel_16.append(results['fused_kernel_16'])

        cumsum.append(results['cumsum'])
        cumsum_kernel.append(results['cumsum_kernel'])
        block_cumsum_kernel.append(block_results['cumsum_kernel_block'])

    plt.plot(chunk_sizes, cumsum, label='CumSum')
    plt.plot(chunk_sizes, cumsum_kernel, label='CumSum Kernel')
    plt.plot(chunk_sizes, block_cumsum_kernel, label='Carved Block CumSum Kernel')

    # plt.plot(chunk_sizes, fused_scan, label='Fused Scan')
    # plt.plot(chunk_sizes, fused_kernel_hadamard, label='Fused Hadamard Kernel')
    # plt.plot(chunk_sizes, fused_kernel_16, label='Fused Tensor Kernel')

    plt.xlabel('Chunk Size')
    plt.ylabel('Time (ms)')
    plt.title('Fusing “inside” chunk')
    plt.legend()

    plt.savefig('/nlp/scr/yusun/data/karan/ttt-gpt/pallas/results/m1_fused_kernel_forward_blocks.png', dpi=300)

def plot_gradient():
    file_path = '/nlp/scr/yusun/data/karan/ttt-gpt/pallas/results/m1_gradient_kernel_forward.pth'
    data = torch.load(file_path)

    contiguous_file_path = '/nlp/scr/yusun/data/karan/ttt-gpt/pallas/results/m1_gradient_kernel_forward_contiguous.pth'
    contiguous_data = torch.load(contiguous_file_path)

    plt.plot(data['sequence_length'], data['scan'], label='Baseline')
    plt.plot(data['sequence_length'], data['kernel_16'], label='Tensor Kernel')
    plt.plot(data['sequence_length'], data['kernel_hadamard'], label='Hadamard Kernel')
    plt.plot(contiguous_data['sequence_length'], contiguous_data['kernel_hadamard'], label='Hadamard Kernel (Contiguous Trick)')    
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (ms)')
    plt.title('Fusing “between” chunks (CS = 1)')
    plt.legend()

    plt.savefig('/nlp/scr/yusun/data/karan/ttt-gpt/pallas/results/m1_gradient_kernel_forward.png', dpi=300)

if __name__ == "__main__":
    plot_chunks()
    # plot_gradient()