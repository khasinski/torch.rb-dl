require "fiddle"

module TorchDL
  module CUDA
    class << self
      def synchronize
        return unless Torch::CUDA.available?

        @cudart ||= load_cudart
        return unless @cudart

        # cudaDeviceSynchronize returns cudaError_t (0 = success)
        @cuda_device_synchronize ||= Fiddle::Function.new(
          @cudart["cudaDeviceSynchronize"],
          [],
          Fiddle::TYPE_INT
        )
        @cuda_device_synchronize.call
      end

      def current_device
        return -1 unless Torch::CUDA.available?

        @cudart ||= load_cudart
        return -1 unless @cudart

        @cuda_get_device ||= Fiddle::Function.new(
          @cudart["cudaGetDevice"],
          [Fiddle::TYPE_VOIDP],
          Fiddle::TYPE_INT
        )

        device_ptr = Fiddle::Pointer.malloc(Fiddle::SIZEOF_INT)
        @cuda_get_device.call(device_ptr)
        device_ptr[0, Fiddle::SIZEOF_INT].unpack1("i")
      end

      def set_device(device_id)
        return unless Torch::CUDA.available?

        @cudart ||= load_cudart
        return unless @cudart

        @cuda_set_device ||= Fiddle::Function.new(
          @cudart["cudaSetDevice"],
          [Fiddle::TYPE_INT],
          Fiddle::TYPE_INT
        )
        @cuda_set_device.call(device_id)
      end

      private

      def load_cudart
        # Try common CUDA runtime library paths
        paths = [
          "libcudart.so",
          "libcudart.so.12",
          "libcudart.so.11",
          "/usr/local/cuda/lib64/libcudart.so",
          "/usr/lib/x86_64-linux-gnu/libcudart.so"
        ]

        paths.each do |path|
          begin
            return Fiddle.dlopen(path)
          rescue Fiddle::DLError
            next
          end
        end

        nil
      end
    end
  end
end
