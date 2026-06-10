# PRISM: Sequence-Independent Protein Domain Detection and Classification


## Authors & Affiliations

**Abel Tan** and **Henning Seedorf**  
*Department of Biological Sciences, National University of Singapore*

*Funded by Temasek Life Sciences Laboratory Limited until funding was cut on 30 January 2026. Further development was self-funded by Abel Tan and/or gratuitously supported by NUS IT and CBIS.

## Overview

**PRISM** (Protein Record Identification & Structural Mapping) is a unified computational framework that integrates sequence-independent domain segmentation and functional classification into a single, high-throughput inference step.

Traditional structural parsing tools rely heavily on sequence alignments and static structural databases (like CATH or SCOP), which biases them toward well-characterized bacterial and eukaryotic folds. PRISM bypasses these limitations by reframing structural parsing as a **2D visual object detection task**. Utilizing a custom-trained YOLOv26 model, PRISM visually interprets geometric patterns within Cα-Cα structural distance matrices, enabling the accurate resolution of complex, novel protein architectures—such as archaeal adhesin-like proteins (ALPs)—that confound standard pipelines.

---

## Key Features

* **One-Shot Segmentation & Classification:** Simultaneously identifies domain boundaries and assigns functional/structural labels in a single step, eliminating the need to decouple parsing from secondary sequence-based searches.
* **Sequence-Independent:** Relies entirely on 3D geometric footprints, avoiding the constraints of sequence homology and static structural databases.
* **Superior Accuracy on Novel Architectures:** Outperforms state-of-the-art tools like Chainsaw and Merizo in detecting and classifying complex, interleaved, or discontinuous domain architectures.
* **Computationally Lean & Robust:** Operates on fixed-resolution image representations, avoiding the GPU memory bottlenecks that often cause structure-based algorithms to fail on exceptionally large proteins (e.g., >3,000 amino acids).
* **Super-Human Precision:** Empirically proven to correct human subjectivity and manual annotation errors, particularly for small domains nested within massive repetitive blocks.

---

## How the Pipeline Works

1. **Structure Parsing:** Extracts 3D coordinates of the Cα backbone from standard input structures (PDB or mmCIF formats). Utilizes Bio.PDB with a robust raw-text fallback parser for non-standard files.
2. **Matrix Generation:** Computes a full pairwise Euclidean distance matrix for the Cα backbone.
3. **Canvas Standardization (Letterboxing):** Centers and symmetrically pads the distance matrices onto a universal square canvas (defined by the maximum sequence length in the dataset) to provide uniform input dimensions without distorting structural geometry.
4. **Image Rendering:** Renders the unified matrices as 2D PNG images utilizing a viridis colormap at a user-defined resolution.
5. **Inference (2D Object Detection):** The custom YOLOv26 model scans the matrix diagonal to detect the distinct, cohesive geometric footprints of internal protein domains, outputting standard bounding boxes (class ID, center X, center Y, width, height).

---

## Installation & Environment Setup

PRISM utilizes Conda to manage framework dependencies and ensure smooth, cross-platform reproducibility.

### 1. Create Environment from File
Create your runtime workspace from the provided configuration file:
```bash
conda env create -f environment.yml

```

### 2. Update Existing Environment

If you are modifying an active environment instead, execute an explicit file-driven update:

```bash
conda env update --file environment.yml --prune

```

### 3. Activate the Workspace

```bash
conda activate yolov26_clone

```

---

## Input & Output Data Configurations

The PRISM training and inference pipeline requires specific input schemas and creates structured directory trees as artifacts.

### Expected Input Files

1. **Protein Structural Repository (`--pdb_root`):**
A folder containing coordinate files in standard `.pdb` or `.cif` format. The files must be named using their unique structural keys (e.g., `A0A123XYZ.pdb` or `A0A123XYZ.cif`).
2. **Domain Annotation Table (`--csv`):**
A comma-separated (`.csv`) or tab-separated (`.tsv`) table tracking your target domains. The pipeline discovers multiclass categorizations dynamically from your metadata. It must contain the following columns:
* `protein_id`: Unique string mapping directly to the target filenames.
* `domain_start`: Integer index marking the first residue of the domain.
* `domain_end`: Integer index marking the terminating residue of the domain.
* `domain_type`: Functional classification label string (e.g., `ABD`, `RBH`).



### Generated Output Directories (`--out`)

The pipeline automatically writes structural dataset matrices formatted for visual object detection networks:

```text
out_directory/
├── data.yaml              # Consolidated dataset channel definitions
├── images/
│   ├── train/             # Standardized letterboxed 2D PNG contact maps
│   └── val/
└── labels/
    ├── train/             # Normalized bounding box coordinates (YOLO format)
    └── val/

```

---

## Execution Guide

### Running the Complete Pipeline via CMD

To execute dataset serialization, train your network, run predictions, and run multi-threshold statistical verification, use the pipeline execution script.

Run the script by targeting your interpreter path explicitly:

```cmd
conda run -n yolov26_clone python PRISM_CATH_letterboxed_multiclass_domains_docstrings.py --csv "/path/to/annotations.csv" --pdb_root "/path/to/structures" --out "/path/to/output_dataset" --epochs 100 --batch 8 --weights yolo26n.pt

```

### Advanced Pipeline Control Flags

Tailor runtime optimization using these diagnostic parameters:

* `--force_build`: Force updates or recreates all intermediate structural PNG maps and text box tracks, even if they match existing files.
* `--no-resume`: Wipes out prior checkpoints and dataset files to restart tasks cleanly from scratch.
* `--eval_only`: Disables asset generation, modeling passes, and predictions to review existing confusion matrices.


---

## Performance Benchmarks

PRISM was rigorously validated on a curated dataset of archaeal adhesin-like proteins (ALPs).

**Segmentation (Single-Class Evaluation):**
PRISM achieved superior performance compared to existing segmentation methods:

* **PRISM:** F1 = 0.919 (Precision: 0.948, Recall: 0.892)
* **Chainsaw:** F1 = 0.843 (Precision: 0.907, Recall: 0.787)
* **Merizo:** F1 = 0.804 (Precision: 0.870, Recall: 0.747)

**One-Shot Classification:**
The framework demonstrates near-perfect discrimination for canonical ALP domains:

* **Archaeal Big Domains (ABD):** F1 = 0.988
* **Right-Handed β-Helical (RBH) Domains:** F1 = 0.988

---

