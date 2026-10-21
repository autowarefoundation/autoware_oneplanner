# Contributing

Contributions are welcome. This repository follows the
[Autoware contribution guidelines](https://autowarefoundation.github.io/autoware-documentation/main/contributing/).

- **Issues:** describe the problem with the ROS distro, Autoware version, GPU, CUDA and TensorRT
  versions, and the node's log and `inference_status` diagnostic.
- **Pull requests:** one logical change per pull request, with a
  [Conventional Commits](https://www.conventionalcommits.org/) title
  (`fix(autoware_oneplanner): ...`). Sign off every commit
  ([DCO](https://developercertificate.org/), `git commit -s`).
- **Checks:** run `pre-commit run --all-files` and the package tests before opening a pull
  request:

  ```bash
  colcon build --packages-up-to autoware_oneplanner
  colcon test --packages-select autoware_oneplanner && colcon test-result --verbose
  ```

  The host tests run on any machine; the GPU tests need CUDA, TensorRT, spconv and a CUDA
  device, and skip themselves without one.

- **Model interface changes:** a change to what the node feeds a model must update
  [docs/model_contract.md](autoware_oneplanner/docs/model_contract.md) in the same pull
  request, and a change that alters the inputs of an existing convention needs a new
  `input_convention`, so that models trained on the old inputs are refused instead of fed
  something else.
