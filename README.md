# AMID-IF: Template-Assisted Protein Inverse Folding

AMID-IF predicts amino acid sequences from target protein backbone structures, using reference template structures, sequences, and residue alignments to guide prediction. The implementation includes separate target and template encoders, TM-score-weighted feature fusion, alignment-based cross-attention, and time-conditioned masked sequence modeling with iterative sampling.

This directory contains the model and training and evaluation scripts. Both scripts consume preprocessed `.pt` datasets; they do not accept raw PDB files directly.

> **Implementation status:** Datasets, pretrained checkpoints, `requirements.txt`, and the previously documented `template/findTemplate.py` are not included. In `model/AMID_IF_model.py`, `import mathAMID_IF` is inconsistent with the subsequent calls to `math.log` and `math.cos`, and no `mathAMID_IF` module is provided. Review and replace this import with `import math` before running the scripts. This README documents the current interfaces; the complete workflow has not been validated end to end.

## 1. Directory Structure

```text
code/
├── README.md
├── AMID_IF_train.py          # Training, validation, and checkpoint loading/saving
├── AMID_IF_test.py           # Per-protein sampling, evaluation, and sequence export
└── model/
    ├── AMID_IF_model.py      # AMID_IF, target/template networks, and aligned cross-attention
    ├── AMID_IF_util.py       # Features, graph layers, losses, and optimizer utilities
    └── utils.py           # Datasets, dynamic batching, and PDB/alignment parsing
```

## 2. Environment and Dependencies

No pinned dependency versions or validated environment specification are included. The source imports the following third-party packages:

| Package | Purpose |
| --- | --- |
| `torch` | Models, tensor operations, training, and checkpoint I/O |
| `numpy` | Coordinate and sequence processing |
| `tqdm` | Progress bars |
| `pandas` | Tabular data loading |
| `scikit-learn` | Dataset splitting utilities |
| `biopython` | PDB structure parsing |
| `python-dateutil` | Imported by `model/utils.py` |
| `sympy` | Imported by `AMID_IF_test.py` |

Install the dependencies in an isolated Python environment:

```bash
python -m pip install torch numpy tqdm pandas scikit-learn biopython python-dateutil sympy
```

GPU training requires a CUDA-enabled PyTorch build compatible with your system's GPU driver. The generic command above does not guarantee GPU support on your machine. Check the installed environment with:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"
```

Both scripts default to `cuda:1`, the second GPU. Use `cuda:0` on a machine with one GPU. The scripts select the CPU when CUDA is unavailable, but they contain CUDA AMP calls, and the complete CPU workflow has not been verified.

## 3. Data Preparation

### 3.1 Input Format

Prepare separate `.pt` files for training, validation, and testing. Each file should deserialize into a list of samples. A sample contains a target protein record under `target` and a dictionary of template records under `pairs`:

```python
dataset = [
    {
        "target": target_record,
        "pairs": {
            "template_1": template_record_1,
            "template_2": template_record_2,
        },
    },
    # Additional samples...
]
```

The following illustrates the fields for a single-chain sample. Replace all placeholders, example sequences, scores, and indices with actual data before use:

```python
target_record = {
    "name": "target_001",
    "seq": "ACDE...",                  # Full sequence across all chains
    "num_of_chains": 1,
    "masked_list": ["A"],             # Chains to design
    "visible_list": [],               # Chains with known sequences
    "seq_chain_A": "ACDE...",
    "coords_chain_A": ...,            # [L, 4, 3]; atom order: N, CA, C, O
}

template_record_1 = {
    "name": "template_1",
    "seq": "ACDF...",
    "num_of_chains": 1,
    "masked_list": ["A"],
    "visible_list": [],
    "seq_chain_A": "ACDF...",
    "coords_chain_A": ...,            # [L_template, 4, 3]
    "TM_score": 0.8,                  # Replace with the structural alignment score
    "Align_idx": [(0, 0), (1, 1)],     # Zero-based (target index, template index) pairs
    "Align_len": 2,
    "u": ...,                         # 3 x 3 rotation matrix
    "t": ...,                         # Translation vector of length 3
}
```

Data conventions:

- Each chain requires matching `seq_chain_<chain_id>` and `coords_chain_<chain_id>` fields. The sequence length must match the first coordinate dimension.
- The sequence alphabet is `ACDEFGHIKLMNPQRSTVWYX`. Other characters are mapped to `X` during feature construction.
- A target residue is valid only when all N, CA, C, and O coordinates are finite. Template validity checks only CA coordinates, but template coordinate arrays must still have shape `[L, 4, 3]`.
- `Align_idx` must refer to the residue order actually passed to the model. Feature construction currently shuffles chain order without remapping alignment indices; this needs to be addressed before using multichain samples.
- The training script reads template `u` and `t` fields, although neither is passed to the current model call. `Align_len` is metadata produced by the parsing utilities.
- `pairs` may be empty; the implementation includes a branch for samples without templates. Samples in a training batch must have the same number of templates, which `MultiStruct_Loader` handles by grouping samples automatically.
- The training and validation dataset class retains only samples whose target sequence length is **less than 2048**. The `--max_protein_length` argument currently does not override this hardcoded limit.

### 3.2 Template Retrieval and Preprocessing

The earlier documentation described retrieving templates with Foldseek and generating `.pt` files through `template/findTemplate.py`. These tools and databases are not included in this directory. Training and evaluation do not invoke Foldseek when using existing preprocessed datasets.

If you only have raw PDB files, first supply a template retrieval and structural alignment workflow, then construct the records described above. `model/utils.py` provides parsing utilities such as `CATHloader_pdb`, `Dual_dataset`, `build_nested_dict`, and `parse_DualPDBs`, but no standalone preprocessing entry point connects them in this directory.

## 4. Training

Run the following commands from the `code` directory. Replace the example paths with your actual dataset locations. Commands are written on a single line for use in either PowerShell or Bash.

### 4.1 Train from Scratch

```bash
python AMID_IF_train.py --train_dataset_path "data/train.pt" --valid_dataset_path "data/valid.pt" --path_for_outputs "outputs/AMID_IF" --device_set cuda:0 --num_epochs 60 --batch_size 8000
```

The default data paths refer to the original development environment and should be overridden explicitly. After resolving the model import issue and installing dependencies, list all arguments with:

```bash
python AMID_IF_train.py --help
```

| Argument | Default | Description |
| --- | --- | --- |
| `--train_dataset_path` | Original development path | Training dataset `.pt` file |
| `--valid_dataset_path` | Original development path | Explicitly specify a separate validation dataset `.pt` file |
| `--path_for_outputs` | `train_outputs/CATH_PDB_with_maskCA_Top2TMration_all` | Log and checkpoint directory |
| `--device_set` | `cuda:1` | Compute device |
| `--num_epochs` | `60` | Total epoch count at which training stops |
| `--batch_size` | `8000` | Budget for the sum of target and template sequence lengths per batch, rather than a protein count; an individual sample exceeding the budget can still form its own batch |
| `--hidden_dim` | `128` | Hidden feature dimension |
| `--num_encoder_layers` | `3` | Encoder depth; currently also used to set decoder depth |
| `--num_decoder_layers` | `3` | Declared but not used when constructing the model in the script |
| `--num_neighbors` | `48` | Target graph neighbor count; the template branch uses a fixed value of 4 |
| `--dropout` | `0.1` | Dropout probability |
| `--backbone_noise` | `0.02` | Backbone coordinate noise magnitude during training |
| `--gradient_norm` | `3.0` | Gradient clipping threshold; nonpositive values disable clipping |
| `--save_model_every_n_epochs` | `5` | Periodic checkpoint interval |
| `--debug` | Disabled | Sets the stopping epoch to 2 and batch budget to 2000; does not reduce the dataset size |

### 4.2 Continue Training from a Checkpoint

```bash
python AMID_IF_train.py --train_dataset_path "data/train.pt" --valid_dataset_path "data/valid.pt" --path_for_outputs "outputs/AMID_IF" --device_set cuda:0 --previous_checkpoint "outputs/AMID_IF/model_weights/epoch_last.pt" --num_epochs 100
```

This loads the model, optimizer state, and completed epoch count, then trains up to epoch 100. The current implementation does not restore the Noam learning-rate scheduler's internal `_step`, so continuation is not guaranteed to match uninterrupted training. See Section 7.

### 4.3 Fine-Tune from Model Weights

```bash
python AMID_IF_train.py --train_dataset_path "data/finetune_train.pt" --valid_dataset_path "data/finetune_valid.pt" --path_for_outputs "outputs/finetune" --device_set cuda:0 --previous_checkpoint "outputs/AMID_IF/model_weights/best_model.pt" --reload_model_only --num_epochs 20
```

`--reload_model_only` resets the optimizer, epoch count, and step count. Model weights are loaded with strict matching, so the architecture must match the checkpoint.

### 4.4 Training Outputs

```text
outputs/AMID_IF/
├── log.txt
└── model_weights/
    ├── best_model.pt            # Checkpoint with the highest validation accuracy
    ├── epoch_last.pt            # Checkpoint after the latest training epoch
    └── epoch<E>_step<S>.pt      # Periodic checkpoint
```

The log records epochs, steps, elapsed time, and training/validation perplexity and accuracy. Checkpoints contain model and optimizer states, epoch and step counts, neighbor count, noise level, and best validation accuracy. They do not include the complete experiment configuration.

The smoothed training loss applies to valid residues selected by random masking. Training perplexity and accuracy are logged over all valid residues, whereas validation metrics cover valid, randomly masked residues in chains designated for design. These metrics therefore use different residue subsets. Calling `model.eval()` does not disable random masking inside `forward()`.

## 5. Evaluation and Sequence Output

`AMID_IF_test.py` is configured through its internal `MyArgs` class and **does not parse command-line arguments**. Edit the following fields before running it:

```python
self.device_set = "cuda:0"
self.path_testset = "data/test.pt"
self.path_for_outputs = "outputs/AMID_IF/test"
self.model_path = "outputs/AMID_IF/model_weights/best_model.pt"
```

Ensure that model settings, including `hidden_dim` and `num_encoder_layers`, match training. The evaluation script also uses `num_encoder_layers` to set decoder depth and does not automatically recover the full model configuration from the checkpoint.

```bash
python AMID_IF_test.py
```

Evaluation processes one protein at a time; `MyArgs.batch_size` does not control batching. The script sets `num_samples = 1` and sampling temperature `temperature = 0.01`. `model.sample()` uses 10 iterations by default. Modify the corresponding source code to change these settings.

Each target produces a `<target_name>.txt` file containing its name, original sequence, predicted sequence, recovery, perplexity, template identifiers, and TM-scores. The final console output reports the checkpoint name, `test_perp`, and `test_rec`. Use unique target names, since duplicate names write to the same output file.

## 6. Metrics

- **Sequence recovery:** The fraction of predicted amino acids matching the reference sequence. Aggregate evaluation weights valid residues in chains designated for design; it is not an unweighted average of per-protein recovery rates.
- **Perplexity:** The exponential of the mean negative log-likelihood over evaluated residues, computed as `exp(mean NLL)`.

Evaluation first generates a sequence with `model.sample()`, then passes that sequence to `model.forward()`, which applies another random mask. The exported `Sampled seq`, exported `Recovery`, and console `test_rec` actually refer to the `argmax` predictions from this second forward pass, **not the original sampled sequence and its recovery**. Recovery of the original sampled sequence is computed inside the loop but is not exported as the result file's `Recovery`. Keep these definitions distinct when interpreting results.

## 7. Known Implementation Limitations

The following observations come from source inspection and should be addressed when running or reproducing experiments:

| Item | Current behavior and required action |
| --- | --- |
| Model import | Review and replace `import mathAMID_IF` in `AMID_IF_model.py` with `import math`; no matching custom module is included. |
| Data and checkpoints | Datasets, checkpoints, and the template generation script must be supplied separately. Replace paths from the original development environment. |
| Mixed precision | `--mixed_precision` uses `type=bool`, so the string `False` does not reliably disable it. The non-AMP branch also does not unpack the model's two-element return value. Fix both issues before switching precision modes. |
| Training continuation | The loaded `total_step` is not assigned to the optimizer wrapper's `_step`. AMP scaler and random-number generator states are not saved either. Exact continuation requires additional state restoration. |
| PyTorch data loading | Training does not specify `weights_only` in `torch.load()`, whereas evaluation explicitly sets it to `False`. If deserialization restrictions reject datasets containing NumPy or other objects, review the loading options. Use full-object deserialization only for trusted files. |
| Multichain alignments | Target and template chains are shuffled independently without updating `Align_idx`. Remap alignment indices or preserve chain order before using multichain templates. |
| Template fusion | The encoder incorporates all templates, but decoder cross-attention currently uses only the first template. Template dictionary order affects this choice. |
| Result interpretation | Validation uses random masking, and evaluation exports predictions from a second forward pass. Standardize random seeds, dataset splits, and metric definitions before comparing methods or reporting results. |

This README was prepared from the source files in this directory. Training, sampling, and evaluation have not been validated with real datasets and checkpoints.
