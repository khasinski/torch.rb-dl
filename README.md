# torch-dl

Multi-GPU training for [torch.rb](https://github.com/ankane/torch.rb). Split batches across multiple GPUs automatically.

## Installation

Add to your Gemfile:

```ruby
gem "torch-dl"
```

## Usage

### Basic Usage

```ruby
require "torch_dl"

# Create your model on the first GPU
model = MyModel.new.to("cuda:0")

# Wrap with DataParallel
dp_model = DataParallel.new(model, device_ids: [0, 1])

# Training loop
optimizer.zero_grad
output = dp_model.call(input)
loss = criterion.call(output, target)
loss.backward
optimizer.step
```

### Models That Return Loss

If your model returns a scalar loss (e.g., GPT models returning `[logits, loss]`), use `dp_model.backward` instead of `loss.backward`:

```ruby
optimizer.zero_grad
logits, loss = dp_model.call(input, targets: targets)
dp_model.backward(scale: 1.0)
optimizer.step
```

This is necessary because gathering scalar tensors across devices breaks the autograd graph. The `backward` method calls backward on each replica's loss separately, then reduces gradients to the original module.

### Gradient Accumulation

For gradient accumulation, scale the backward pass:

```ruby
gradient_accumulation_steps = 4

(0...gradient_accumulation_steps).each do |step|
  logits, loss = dp_model.call(input_batch, targets: targets_batch)
  dp_model.backward(scale: 1.0 / gradient_accumulation_steps)
end
optimizer.step
optimizer.zero_grad
```

## API Reference

### DataParallel

```ruby
DataParallel.new(model, device_ids: nil, output_device: nil, dim: 0)
```

- `model` - The module to parallelize (must be on `cuda:0`)
- `device_ids` - Array of GPU indices to use (default: all available)
- `output_device` - GPU index for output (default: first device)
- `dim` - Dimension to scatter inputs along (default: 0, batch dimension)

#### Methods

- `call(*inputs, **kwargs)` - Forward pass with automatic scattering/gathering
- `backward(scale: 1.0)` - Backward pass for loss-returning models
- `module` / `wrapped_module` - Access the underlying model
- `parameters` - Access model parameters (for optimizer)
- `state_dict` / `load_state_dict` - Save/load model state
- `train` / `eval` - Set training/evaluation mode

### Low-level Functions

```ruby
# Split tensor across devices
TorchDL.scatter(tensor, ["cuda:0", "cuda:1"], dim)

# Gather tensors to a single device
TorchDL.gather(tensors, "cuda:0", dim)
```

## How It Works

1. **Scatter**: Input batch is split across GPUs
2. **Replicate**: Model is copied to each GPU (cached for performance)
3. **Parallel Apply**: Forward pass runs on each GPU in parallel using threads
4. **Gather**: Outputs are collected back to the output device

## Requirements

- Ruby 3.1+
- torch.rb 0.17+
- Multiple CUDA GPUs

## Notes

- Works with stock torch.rb from RubyGems
- For optimal performance, use a torch.rb build with `Torch::CUDA.synchronize` support (ensures CUDA operations complete before gathering)

## License

MIT
