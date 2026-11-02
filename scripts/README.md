# ParticleMan Data Conversion Scripts

This directory contains scripts for converting particle physics data to formats suitable for training the ParticleTransformer model.

## `convert_root_to_h5.py`

Converts ROOT ntuples to padded HDF5 files using **uproot** (no ROOT/PyROOT installation needed). The core logic lives in `src/particleman/data/conversion.py`.

The conversion is driven by a data config (`configs/data/*.yaml`), the same one `ROOTParticleLoader` uses:

- `source.files` / `source.tree_name`: input files (glob patterns allowed) and tree
- `collections`: which branches to read; enabled collections are merged per event in config order
- `particle_id_map` / `default_particle_id`: raw ID (PDG code, jet truth label) to categorical ID
- `preprocessing`: `pt_scale`, `pt_cut`, `eta_cut`, `max_particles`

Each event in the output equals `ROOTParticleLoader.get_event(i)` with `shuffle_particles=false`. Particles are not shuffled on disk; turn shuffling on when training on the output.

Each input `<name>.root` is written to `<output_dir>/<name>.h5`, so `file_event_label_map` patterns (e.g. `ttbar*`) still match the converted files.

### Usage

```bash
# Convert every file in the bbllv08 config
python scripts/convert_root_to_h5.py data=bbllv08 ++output_dir=/path/to/h5

# Options
python scripts/convert_root_to_h5.py data=bbllv08 ++output_dir=h5 \
    ++step_size=10000 \            # entries read per uproot chunk
    ++overwrite=true \             # default: skip outputs that already exist
    ++skip_empty=true \            # drop events with no particles after cuts
    data.max_events=1000 \         # per input file
    data.preprocessing.max_particles=100 \
    data.source.files='[a.root,b.root]'
```

Missing branches raise an error before anything is written. Use `explore_root_file.py` to find branch names.

### Output Format

**Datasets:**
- `pt`: Transverse momentum in GeV (after `pt_scale`), zero-padded [n_events, max_particles] float32
- `eta`: Pseudorapidity [n_events, max_particles] float32
- `phi`: Azimuthal angle in radians [n_events, max_particles] float32
- `particle_id`: Categorical particle ID, padded with `default_particle_id` [n_events, max_particles] int32
- `n_particles`: Particles after cuts, before truncation [n_events] int32
- `event_id`: Entry index in the source tree [n_events] int64

**Attributes:** `n_events`, `source_file`, `tree_name`, `max_particles`, `pt_scale_applied`, `pt_cut`, `eta_cut`, `particle_id_map`, `default_particle_id`, `collections`.

### Training on the Output

Use `configs/data/preprocessed_h5.yaml`, which has identity ID mapping and no cuts or scaling:

```bash
python scripts/run.py data=preprocessed_h5 \
    data.source.files='[/path/to/h5/*.h5]' \
    data.preprocessing.max_particles=<same as conversion> \
    data.preprocessing.shuffle_particles=true
```

To train a classifier, add the same `file_event_label_map` and `n_event_types` as the source config.
