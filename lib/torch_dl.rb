require "torch"
require_relative "torch_dl/version"
require_relative "torch_dl/ext"
require_relative "torch_dl/nn/parallel/replicate"
require_relative "torch_dl/nn/parallel/parallel_apply"
require_relative "torch_dl/nn/parallel/data_parallel"

module TorchDL
  class Error < StandardError; end

  class << self
    def scatter(input, devices, dim = 0)
      TorchDL::Ext.scatter(input, devices, dim)
    end

    def gather(inputs, target_device, dim = 0)
      TorchDL::Ext.gather(inputs, target_device, dim)
    end
  end
end

# Convenience alias
DataParallel = TorchDL::NN::DataParallel
