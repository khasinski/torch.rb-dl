module TorchDL
  module NN
    module Replicate
      class << self
        # Replicates a module on multiple devices.
        #
        # @param network [Torch::NN::Module] The module to replicate
        # @param devices [Array<String>] List of device strings (e.g., ["cuda:0", "cuda:1"])
        # @return [Array<Torch::NN::Module>] List of module replicas, one per device
        #
        # Note: The first device uses the original network (not a copy) to ensure
        # gradients flow back to the original parameters during backward pass.
        def replicate(network, devices)
          devices = devices.map { |d| d.is_a?(String) ? d : "cuda:#{d}" }

          # Single device - just return the network (already on correct device)
          return [network] if devices.size == 1

          # Get the state dict once for creating replicas
          state_dict = network.state_dict

          # Create replicas - first device uses original network for gradient flow
          devices.each_with_index.map do |device, idx|
            if idx == 0
              # First device: use the original network to maintain gradient connection
              network
            else
              # Other devices: create independent replicas
              replica = deep_copy_module(network)

              # Copy state dict tensors to the target device
              # Filter to only include keys that exist in the replica
              replica_keys = replica.state_dict.keys
              device_state = state_dict.select { |k, _| replica_keys.include?(k) }
                                       .transform_values { |t| t.to(device) }
              replica.load_state_dict(device_state)
              replica.to(device)
              replica.train(network.instance_variable_get(:@training))
              replica
            end
          end
        end

        private

        # Creates a deep copy of a module structure
        def deep_copy_module(mod)
          # Check for custom replication hook first
          return mod.class._replicate(mod) if mod.class.respond_to?(:_replicate)

          # Handle container modules
          return copy_sequential(mod) if mod.is_a?(Torch::NN::Sequential)
          return copy_module_list(mod) if mod.is_a?(Torch::NN::ModuleList)

          # Try built-in module copiers
          copier = module_copiers[mod.class]
          return copier.call(mod) if copier

          # Fallback to generic copy for custom modules
          copy_custom_module(mod)
        end

        # Lazily initialized registry of module copiers
        def module_copiers
          @module_copiers ||= {
            Torch::NN::Linear => ->(m) { Torch::NN::Linear.new(m.in_features, m.out_features, bias: has_bias?(m)) },
            Torch::NN::Conv1d => ->(m) { copy_conv(Torch::NN::Conv1d, m) },
            Torch::NN::Conv2d => ->(m) { copy_conv(Torch::NN::Conv2d, m) },
            Torch::NN::Conv3d => ->(m) { copy_conv(Torch::NN::Conv3d, m) },
            Torch::NN::BatchNorm1d => ->(m) { copy_batch_norm(Torch::NN::BatchNorm1d, m) },
            Torch::NN::BatchNorm2d => ->(m) { copy_batch_norm(Torch::NN::BatchNorm2d, m) },
            Torch::NN::BatchNorm3d => ->(m) { copy_batch_norm(Torch::NN::BatchNorm3d, m) },
            Torch::NN::LayerNorm => ->(m) {
              Torch::NN::LayerNorm.new(m.instance_variable_get(:@normalized_shape),
                               eps: m.instance_variable_get(:@eps),
                               elementwise_affine: m.instance_variable_get(:@elementwise_affine))
            },
            Torch::NN::Embedding => ->(m) {
              Torch::NN::Embedding.new(m.instance_variable_get(:@num_embeddings),
                               m.instance_variable_get(:@embedding_dim),
                               padding_idx: m.instance_variable_get(:@padding_idx))
            },
            Torch::NN::Dropout => ->(m) { Torch::NN::Dropout.new(p: m.instance_variable_get(:@p)) },
            Torch::NN::Dropout2d => ->(m) { Torch::NN::Dropout2d.new(p: m.instance_variable_get(:@p)) },
            Torch::NN::Dropout3d => ->(m) { Torch::NN::Dropout3d.new(p: m.instance_variable_get(:@p)) },
            Torch::NN::LSTM => ->(m) { copy_rnn(Torch::NN::LSTM, m) },
            Torch::NN::GRU => ->(m) { copy_rnn(Torch::NN::GRU, m) },
            Torch::NN::RNN => ->(m) { copy_rnn(Torch::NN::RNN, m) },
            Torch::NN::ReLU => ->(_) { Torch::NN::ReLU.new },
            Torch::NN::GELU => ->(_) { Torch::NN::GELU.new },
            Torch::NN::Tanh => ->(_) { Torch::NN::Tanh.new },
            Torch::NN::Sigmoid => ->(_) { Torch::NN::Sigmoid.new },
            Torch::NN::Identity => ->(_) { Torch::NN::Identity.new },
            Torch::NN::Softmax => ->(m) { Torch::NN::Softmax.new(dim: m.instance_variable_get(:@dim)) },
            Torch::NN::LogSoftmax => ->(m) { Torch::NN::LogSoftmax.new(dim: m.instance_variable_get(:@dim)) },
          }
        end

        def has_bias?(mod)
          !mod.instance_variable_get(:@bias).nil?
        end

        def copy_conv(klass, mod)
          klass.new(mod.in_channels, mod.out_channels, mod.kernel_size,
                    stride: mod.stride, padding: mod.padding, dilation: mod.dilation,
                    groups: mod.groups, bias: has_bias?(mod), padding_mode: mod.padding_mode)
        end

        def copy_batch_norm(klass, mod)
          klass.new(mod.num_features, eps: mod.eps, momentum: mod.momentum,
                    affine: mod.affine, track_running_stats: mod.track_running_stats)
        end

        def copy_rnn(klass, mod)
          klass.new(mod.input_size, mod.hidden_size, num_layers: mod.num_layers,
                    bias: mod.bias, batch_first: mod.batch_first,
                    dropout: mod.dropout, bidirectional: mod.bidirectional)
        end

        def copy_sequential(mod)
          Torch::NN::Sequential.new(*mod.children.map { |child| deep_copy_module(child) })
        end

        def copy_module_list(mod)
          Torch::NN::ModuleList.new(mod.map { |child| deep_copy_module(child) })
        end

        def copy_custom_module(mod)
          klass = mod.class
          children = mod.named_children.to_h

          if children.any?
            # Module has submodules - create structural copy
            replica = klass.allocate
            replica.send(:initialize_module_state)
            copy_instance_state(mod, replica)
            children.each do |name, child|
              replica.instance_variable_set("@#{name}", deep_copy_module(child))
            end
            replica
          else
            # Leaf module - try clone
            mod.clone
          end
        rescue => e
          raise ArgumentError, "Cannot replicate #{klass}. " \
            "Implement #{klass}._replicate(mod) class method. Error: #{e.message}"
        end

        def copy_instance_state(src, dst)
          src.instance_variables.each do |ivar|
            next if %i[@parameters @buffers @modules @training @non_persistent_buffers_set].include?(ivar)
            val = src.instance_variable_get(ivar)
            next if val.is_a?(Torch::Tensor) || val.is_a?(Torch::NN::Module)
            dst.instance_variable_set(ivar, val)
          end
        end
      end
    end
  end
end

# Add helper method to Module for initializing state (if not already defined)
module Torch
  module NN
    class Module
      private

      def initialize_module_state
        @training = true
        @parameters = {}
        @buffers = {}
        @modules = {}
        @non_persistent_buffers_set = Set.new
      end unless private_method_defined?(:initialize_module_state)
    end
  end
end
