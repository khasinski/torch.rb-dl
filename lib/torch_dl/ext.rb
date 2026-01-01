module TorchDL
  # Pure Ruby implementation of scatter/gather operations
  module Ext
    class << self
      # Splits a tensor across devices
      def scatter(input, devices, dim = 0)
        chunks = input.chunk(devices.size, dim)
        chunks.each_with_index.map do |chunk, i|
          chunk.to(devices[i])
        end
      end

      # Gathers tensors from multiple devices onto a single device
      def gather(inputs, target_device, dim = 0)
        on_target = inputs.map { |t| t.to(target_device) }
        Torch.cat(on_target, dim)
      end
    end
  end
end
