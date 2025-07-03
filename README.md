# Image Analysis: Determining Page Ordering of the Voynich Manuscript

This repository contains the implementation of a research project exploring the use of Siamese Neural Networks to investigate the physical ordering of pages in historical manuscripts, with a focus on the enigmatic **Voynich Manuscript**.

## Overview

The Voynich Manuscript is a mysterious illustrated codex from the early 15th century, written in an unknown script. Scholars suspect that some of its pages might not be in their original order due to rebinding events in its history.

This project investigates whether **visual features** such as liquid stains can help identify adjacent pages and possible reorderings in the manuscript.

## Methodology

We developed a Siamese Network trained to measure **visual similarity** between pairs of manuscript pages.

### Approach

- Trained the model on three digitized historical manuscripts with known page orders:
  - _Ricettario_ (ca. 1530)
  - _Shahanshahnamah_ (ca. 1815–1850)
  - _Sermons_ (ca. 1500–1600)
- Preprocessed images to remove noise and created positive (adjacent) and negative (non-adjacent) page pairs.
- Used a **contrastive loss function** to learn embeddings where adjacent pages are close together, and non-adjacent pages are far apart.

### Model

A Siamese architecture with twin CNN branches to extract features, followed by a Euclidean distance computation to quantify similarity.
