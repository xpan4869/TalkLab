# TalkLab Take-Home Assignment: Conversational Analysis Pipeline

This repository contains the code and data outputs for the TalkLab take-home assignment, including the computational pipeline (Question 2), the conversational sequence analysis (Question 3), a zipped video answer file, and supplemental written answers.

## Repository Structure & Execution Order

**`data/`**
Contains all original, unmodified data files.
Processed outputs generated throughout the pipeline are stored in **`data/processed/`**.

**`scripts/`**
Includes all preprocessing and analysis scripts.

* **`scripts/convo-sequence_analysis/`**

  * `sequence-analysis.R` is the main analysis script.
  * Annotated results are available in `convo-sequence_analysis.html` and `convo-sequence_analysis.Rmd`.

* **`scripts/feature-extraction/`**
  Implements the full feature extraction workflow used to answer Question 2.

**`Pan_Presentation.mp4.zip`**
Zipped archive containing the video answer file (`Presentation.mp4`).

**`take-home-answer/`**
Contains the PDF with written explanations that supplement the video answers.