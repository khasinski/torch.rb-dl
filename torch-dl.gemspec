require_relative "lib/torch_dl/version"

Gem::Specification.new do |spec|
  spec.name = "torch-dl"
  spec.version = TorchDL::VERSION
  spec.authors = ["Chris Hasinski"]
  spec.email = ["krzysztof.hasinski@gmail.com"]

  spec.summary = "Multi-GPU training for torch.rb"
  spec.description = "DataParallel and distributed training utilities for torch.rb. Split batches across multiple GPUs automatically."
  spec.homepage = "https://github.com/khasinski/torch-rb-dl"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.1"

  spec.metadata["homepage_uri"] = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["changelog_uri"] = "#{spec.homepage}/blob/master/CHANGELOG.md"

  spec.files = Dir["*.{md,txt}", "{lib}/**/*"]
  spec.require_paths = ["lib"]

  spec.add_dependency "torch-rb", ">= 0.17"
end
