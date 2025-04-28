
# Deep Learning Final Project: Locating Bacterial Flagellar Motors in 3D Tomograms

## Introduction - What is the Problem and Why Does it Matter?

Bacterial flagellar motors are remarkable biological machines that enable bacteria to move through their environment. These complex molecular structures, essentially functioning as nanoscale rotary engines, are critical components of bacterial motility and play significant roles in pathogenesis, biofilm formation, and host interactions. However, locating these motors in 3D tomographic reconstructions has traditionally been a manual, time-consuming task requiring expert annotation.

The challenge addressed in this project is the development of an automated algorithm to accurately identify the presence and 3D coordinates of flagellar motors in tomographic reconstructions. Automating this process would accelerate research significantly, enabling scientists to analyze larger datasets more efficiently. This acceleration has far-reaching implications for advancing molecular biology, improving drug development targeting bacterial motility, and enhancing synthetic biology applications.

The specific question this project aims to answer is: Can a deep learning model effectively locate bacterial flagellar motors in 3D tomograms with accuracy comparable to manual annotation?

My approach was deliberately incremental, starting with a minimal viable model and progressively refining it based on performance and computational constraints. This iterative process led to a progressive sampling strategy that balanced computational efficiency with detection accuracy. The results demonstrate that deep learning can successfully automate this task, achieving promising detection results while managing the computational challenges inherent in processing 3D volumetric data.

## Approach - How Did You Address the Question?

### Data

This project utilized the BYU Locating Bacterial Flagellar Motors dataset from Kaggle, consisting of 648 tomograms (73.87 GB total). Each tomogram contained between 300–800 2D slices stored as JPEG images, with expert-annotated motor locations provided as 3D coordinates.

The dataset offered:

- High-resolution tomographic reconstructions where motors are visible,
- Ground truth annotations from experts,
- A diverse set of tomograms with varying image quality and motor positions.

Challenges included the massive size of each tomogram, exceeding available GPU memory for full-volume processing, and the "needle-in-a-haystack" problem of motors occupying small regions within large 3D volumes.

### Methodology

#### Data Processing and Loading

- Progressive Sampling Strategy: A coarse scan (20% of slices) identified candidate regions, followed by targeted sampling (up to 80%) where motors were suspected.
- Dynamic Batch Sizing: Adjusted based on real-time GPU memory.
- Custom TomogramDataGenerator: Loaded slices on-demand, normalized contrast using percentiles, and applied real-time data augmentation (flips, rotations, noise).

Each slice was resized to 224×224 pixels and processed with histogram equalization.

#### Model Architecture

- Compact 3D CNN architecture:
    - Input size (150, 224, 224, 1),
    - Four convolutional blocks (16–128 filters),
    - Global average pooling,
    - Dense layer with 128 neurons and dropout (30%),
    - Final dense layer predicting [confidence, z, y, x].

More complex architectures (3D U-Net, transformers) were explored but were too resource-intensive.

#### Training Approach

- Custom Loss Function: Combined binary cross-entropy (motor presence) and mean squared error (coordinate localization).
- Optimization: Adam optimizer with learning rate scheduling and early stopping.
- Metrics: Detection accuracy and coordinate error.

#### Inference Strategy

- Three-Stage Progressive Sampling:
    1. Coarse scan (20% slices),
    2. Focused resampling in regions of interest,
    3. 3D Non-Maximum Suppression (NMS) to refine detections.

## Analysis and Results - What Did You Find?

### Training Performance

![IMAGEBYU](https://github.com/user-attachments/assets/69e74b86-104e-4bf7-a816-9c7a0fbc5317)

Key observations (see Figure 1):

- Loss Convergence: Training loss dropped rapidly, while validation loss showed volatility between epochs 8–12, reflecting generalization challenges.
- Detection Accuracy: Training accuracy steadily improved; validation accuracy peaked around epochs 8–9 and 13–14 at ~78%.
- Coordinate Error: Decreased steadily from ~0.55 to ~0.32, indicating improved localization precision.
- Learning Rate Impact: Post-epoch 11, validation performance improved following learning rate reduction.

### Pattern Analysis and Model Behavior

- Volumetric Strength: The 3D convolutions captured spatial relationships across slices.
- Confidence Scores: Varied between 0.709–1.119, sensitive to image quality.
- Progressive Sampling: Detected motors effectively while analyzing only 20–30% of slices.
- Impact of NMS: Refined detections by merging nearby candidates (e.g., reducing 81 candidates to 73 in tomo_003acc).

### Performance vs. Computational Trade-offs

This project demonstrated that a simple 3D CNN combined with progressive sampling can achieve accurate motor detection while avoiding the prohibitive costs of full-volume analysis. Strategic data handling enabled success under practical GPU constraints.

### Summary in Context

The results confirm that deep learning can automate the detection of bacterial flagellar motors in 3D tomograms with precision close to manual annotation. Efficiency-focused strategies like progressive sampling are both practical and effective for scaling deep learning approaches in biological imaging.

## Discussion, Conclusions, and Next Steps

### Discussion & Interpretation

This study shows that a 3D CNN with progressive sampling can automate motor detection, significantly accelerating structural biology research. Despite computational constraints, the model maintained high detection quality, although variability in tomogram quality remains a generalization challenge.

### Limitations

Key limitations include:

- Small and homogeneous test set,
- Constraints on model complexity due to available resources,
- Potential variability in manual ground truth annotations,
- Focus on a single structure (flagellar motors).

### Conclusion

The project demonstrates that even relatively simple models, when combined with smart data sampling, can effectively automate challenging 3D detection tasks. The approach achieved ~77% validation accuracy and low coordinate error while maintaining computational efficiency.

### Future Work / Next Steps

Future efforts should:

- Validate on larger, more diverse datasets,
- Explore more complex architectures (e.g., 3D U-Net, transformers),
- Integrate multi-scale analysis and semi-supervised learning,
- Extend detection to multiple structures,
- Improve uncertainty quantification and explainability to enhance trust and interpretability.

Integration into real biological research workflows would provide the strongest validation of this approach's practical utility.
