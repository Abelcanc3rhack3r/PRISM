# PRISM: Sequence-Independent Protein Domain Detection and Classification

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

## Authors & Affiliations

**Abel Tan** and **Henning Seedorf**
*Temasek Life Sciences Laboratory Limited, National University of Singapore*
*Department of Biological Sciences, National University of Singapore*