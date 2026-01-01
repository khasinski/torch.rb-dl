module TorchDL
  # Pure Ruby implementation of scatter/gather operations
  # Uses torch.rb's existing tensor operations
  module Ext
    class << self
      # Splits a tensor across devices
      #
      # @param input [Torch::Tensor] The input tensor to scatter
      # @param devices [Array<String>] Target device strings (e.g., ["cuda:0", "cuda:1"])
      # @param dim [Integer] Dimension along which to split (default: 0)
      # @return [Array<Torch::Tensor>] Array of tensors, one per device
      def scatter(input, devices, dim = 0)
        chunks = input.chunk(devices.size, dim)
        chunks.each_with_index.map do |chunk, i|
          chunk.to(devices[i])
        end
      end

      # Gathers tensors from multiple devices onto a single device
      #
      # @param inputs [Array<Torch::Tensor>] Array of tensors to gather
      # @param target_device [String] Target device string (e.g., "cuda:0")
      # @param dim [Integer] Dimension along which to concatenate (default: 0)
      # @return [Torch::Tensor] Concatenated tensor on target device
      def gather(inputs, target_device, dim = 0)
        on_target = inputs.map { |t| t.to(target_device) }
        Torch.cat(on_target, dim)
      end
    end
  end
end
