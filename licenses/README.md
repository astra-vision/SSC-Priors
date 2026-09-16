# Third-party license texts

This directory holds the verbatim license texts (and copyright notices) for the
third-party software and data this project incorporates or depends on. It
complements the human-readable summary in [`../THIRD-PARTY-NOTICES.txt`](../THIRD-PARTY-NOTICES.txt).

The SSC-Priors **source code** is released under the Apache License 2.0
([`../License.txt`](../License.txt)). The pretrained **weights** and the
**dataset** carry separate terms — see below and the License section of the README.

| File | Component | License |
|------|-----------|---------|
| [`SemCity-MIT.txt`](SemCity-MIT.txt) | SemCity (triplane autoencoder, parts of the data pipeline and training loop) | MIT |
| [`LovaszSoftmax-MIT.txt`](LovaszSoftmax-MIT.txt) | Lovász-Softmax loss (`losses.py`) | MIT |
| [`Cylinder3D-Apache-2.0.txt`](Cylinder3D-Apache-2.0.txt) | Cylinder3D (Lovász-Softmax integration in `losses.py`) | Apache-2.0 |
| [`semantic-kitti-api-MIT.txt`](semantic-kitti-api-MIT.txt) | SemanticKITTI dev-kit (`data/semantic-kitti.yaml`, eval convention) | MIT |
| [`SemanticKITTI-dataset-CC-BY-NC-SA-4.0.txt`](SemanticKITTI-dataset-CC-BY-NC-SA-4.0.txt) | SemanticKITTI **dataset** (training data; governs the released checkpoints) | CC BY-NC-SA 4.0 |

WaffleIron (`tools/generate_waffleiron_labels.py` follows its `eval_kitti.py`) is
Apache-2.0 like this repository, whose [`../License.txt`](../License.txt) holds
the license text.

Runtime Python dependencies (PyTorch, NumPy, SciPy, PyYAML, tqdm,
TensorBoard) are installed by the user via `pip` and are
not redistributed here; their licenses are listed in
[`../THIRD-PARTY-NOTICES.txt`](../THIRD-PARTY-NOTICES.txt).
