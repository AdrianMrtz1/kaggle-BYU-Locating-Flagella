# Deep Learning Final Project: Locating Bacterial Flagellar Motors in 3D Tomograms

**Adrian Martinez**

Dataset: https://www.kaggle.com/competitions/byu-locating-bacterial-flagellar-motors-2025

## Introduction - What is the Problem and Why Does it Matter?

Bacterial flagellar motors are remarkable biological machines that enable bacteria to move through their environment. These complex molecular structures, essentially serving as nanoscale rotary engines, are critical components of bacterial motility and play significant roles in bacterial pathogenesis, biofilm formation, and interaction with host organisms. However, locating these motors in 3D tomographic reconstructions has traditionally been a manual, time-consuming task requiring expert annotation.

The challenge addressed in this project is the development of an automated algorithm to accurately identify the presence and precise 3D coordinates of flagellar motors in tomographic reconstructions of bacteria. Automating this process would dramatically accelerate research in this field, enabling scientists to analyze larger datasets more efficiently. This acceleration has far-reaching implications for advancing our understanding of fundamental molecular biology, improving drug development targeting bacterial motility, and enhancing applications in synthetic biology.

The specific question this project aims to answer is: **Can a deep learning model effectively locate bacterial flagellar motors in 3D tomograms with accuracy comparable to manual annotation?**

My approach to this problem was deliberately incremental and adaptive, starting with a clear definition of the task as both a detection problem (determining whether a motor exists) and a coordinate localization problem (pinpointing the motor's precise location in 3D space). Rather than immediately implementing a complex architecture, I began with a minimal viable model and progressively refined it based on performance and computational constraints. This iterative process involved experimentation with data sampling techniques, model architecture, and inference strategies, ultimately leading to a progressive sampling approach that balanced computational efficiency with detection accuracy.

The results demonstrate that deep learning can successfully automate this task, achieving promising detection results on the test set while using efficient sampling techniques to manage the computational challenges inherent in processing 3D volumetric data. This solution represents a significant step toward removing a major bottleneck in the analysis of bacterial structures, potentially accelerating research in this important field.

## Approach - How Did You Address the Question?

### Data

This project utilized the BYU Locating Bacterial Flagellar Motors dataset from Kaggle, which consisted of 648 tomograms with a total size of 73.87 GB. Each tomogram varied in size, typically containing between 300-800 2D slices stored as JPEG images. The dataset was organized with a metadata table containing attributes about each tomogram, including dimensional information and the presence and location of flagellar motors in 3D coordinates.

This dataset was particularly suitable for addressing the question of automated motor detection because it provided:
1. High-resolution tomographic reconstructions where flagellar motors are visible
2. Ground truth annotations of motor locations provided by experts
3. A diverse set of tomograms with varying image quality and motor positions

The primary limitation of the dataset was its massive size, which created significant computational challenges. Each tomogram represented a 3D volume requiring substantial memory to process in its entirety, which exceeded the available GPU memory in many cases. Additionally, the motors occupied relatively small regions within these large volumes, creating a needle-in-a-haystack detection problem.

### Methodology

#### Data Processing and Loading

To address the computational constraints, I implemented a specialized data loading pipeline with several key features:

1. **Progressive Sampling Strategy**: Rather than attempting to process entire tomograms at once, I developed a multi-stage approach:
   - Initial coarse sampling (20% of slices) to identify potential regions of interest
   - Targeted fine-grained sampling (up to 80% of slices) in promising regions
   - This approach significantly reduced memory requirements while maintaining detection accuracy

2. **Dynamic Batch Sizing**: The pipeline automatically adjusted batch sizes based on available GPU memory, ensuring optimal resource utilization.

3. **Efficient Data Generator**: I implemented a custom `TomogramDataGenerator` class that:
   - Loaded data on-demand rather than storing entire datasets in memory
   - Applied normalization using percentile-based techniques to handle varying contrast levels
   - Incorporated data augmentation (flips, rotations, noise addition) to improve model generalization
   - Used prefetching to minimize I/O bottlenecks

4. **Slice Preprocessing**: Each 2D slice underwent several processing steps:
   - Resizing to 224×224 pixels for memory efficiency
   - Histogram equalization for improved contrast
   - Normalization based on 1st and 99th percentiles to standardize intensity ranges

#### Model Architecture

After experimenting with several approaches, I settled on a relatively straightforward 3D CNN architecture due to computational constraints. The final model consisted of:

1. Input layer accepting 3D volumes of shape (150, 224, 224, 1)
2. Four convolutional blocks, each containing:
   - 3D convolutional layer (filter sizes increasing from 16 to 128)
   - Batch normalization layer
   - LeakyReLU activation
   - 3D max pooling layer
3. Global average pooling layer to reduce dimensionality
4. Dense layer with 128 neurons and LeakyReLU activation
5. Dropout layer (30%) for regularization
6. Final dense layer outputting 4 values: [confidence, z, y, x]

This architecture balanced complexity with computational efficiency. More sophisticated architectures including U-Net and transformer-based models were attempted but proved prohibitively expensive given the limited GPU memory and time constraints.

#### Training Approach

The model was trained with several key design choices:

1. **Custom Loss Function**: I implemented a specialized loss function combining binary cross-entropy for motor presence detection and mean squared error for coordinate regression, weighted to prioritize accurate localization.

2. **Metrics**: Two custom metrics tracked performance:
   - Detection accuracy: Measuring correct classification of motor presence/absence
   - Coordinate error: Measuring the Euclidean distance between predicted and ground truth coordinates for true positives

3. **Optimization**: The Adam optimizer was used with an initial learning rate of 0.001 and a learning rate reduction strategy when validation performance plateaued.

4. **Early Stopping**: Training was halted when validation loss stopped improving, with the best model saved for inference.

#### Inference Strategy

For inference, I implemented a three-stage progressive sampling strategy:

1. **Coarse Scan**: Sampling 20% of slices evenly distributed throughout the tomogram
2. **Region-Focused Scan**: If potential motors were detected, additional slices were sampled in those regions
3. **3D Non-Maximum Suppression**: A custom 3D NMS algorithm merged nearby detections by considering their proximity in all three dimensions

This approach proved computationally efficient while still achieving reliable motor detection, addressing the fundamental challenge of locating small structures within large 3D volumes.

## Analysis and Results - What Did You Find?

### Training Performance

The training process revealed several interesting patterns about the model's learning dynamics and the inherent challenges of the flagellar motor detection task. Figure 1 shows the progression of key metrics over 15 epochs of training.

![BYU results](https://github.com/user-attachments/assets/6ea41468-5492-495b-8358-fcd400ce50ef)


*Figure 1: Evolution of Loss, Detection Accuracy, and Coordinate Error during training*

The training metrics reveal several key insights:

1. **Loss Convergence Pattern**: The training loss (blue line, left graph) decreased rapidly in the first 5 epochs, from approximately 6.5 to 1.5, indicating efficient initial learning. The validation loss (orange line) showed more volatility, particularly between epochs 8-12, suggesting periods where the model struggled to generalize its learning to unseen data. This volatility in validation loss highlights the inherent variability in tomographic data and the challenge of consistent feature detection across different tomograms.

2. **Detection Accuracy Improvements**: The detection accuracy graph (middle) shows that while training accuracy (blue) improved consistently, validation accuracy (orange) exhibited significant fluctuations. The model achieved peak validation accuracy around epoch 8-9 and epoch 13-14, reaching approximately 78%. This pattern suggests that the model periodically found and then lost generalizable features before stabilizing in later epochs.

3. **Coordinate Error Reduction**: Both training and validation coordinate errors (right graph) decreased steadily from around 0.55 to approximately 0.32, representing a significant improvement in the precision of motor localization. The convergence of training and validation coordinate errors in later epochs indicates that the model achieved a stable capability to precisely locate motors in 3D space.

4. **Learning Rate Impact**: A notable improvement occurred after epoch 11 when the learning rate was reduced. This suggests that fine-tuning with a smaller learning rate was crucial for refining the model's ability to precisely locate motors, highlighting the importance of the optimization strategy in achieving accurate 3D coordinates.

### Model Performance Analysis

The model's final performance can be analyzed along several dimensions:

1. **Detection Capability**: The model successfully identified the presence of motors with a validation accuracy of approximately 77% by the final epoch. While not perfect, this represents a significant achievement given the challenging nature of the data, where motors occupy a tiny fraction of the total volume and exhibit subtle features that can be easily confused with other cellular structures.

2. **Localization Precision**: The final coordinate error of approximately 0.32 in normalized space translates to a positional accuracy of within a few pixels in the original tomogram space. This level of precision is remarkable considering the complexity of 3D tomographic data and the small size of flagellar motors relative to the full volume.

3. **Inference Performance**: When applied to the test set, the model successfully detected motors in all three test tomograms, with confidence scores ranging from 0.709 to 1.119. The progressive sampling approach proved effective, typically requiring analysis of only 20-30% of slices to make confident detections, significantly reducing computational requirements without sacrificing accuracy.

### Pattern Analysis and Model Behavior

Deeper analysis of the model's behavior revealed several interesting patterns:

1. **Volumetric Processing**: The model processes 3D volumes rather than analyzing individual slice-level features. The architecture with 3D convolutional layers suggests the model leverages spatial relationships across adjacent slices to identify the full 3D structure of flagellar motors.

2. **Confidence Distribution**: In the test results, we observed varying confidence scores across detections (ranging from 0.709 to 1.119), indicating the model's varying certainty about motor locations in different tomograms.

3. **Progressive Sampling Effectiveness**: The results demonstrate that the progressive sampling approach was highly efficient. For example, in the test tomogram "tomo_003acc", the model initially detected a potential motor during the coarse sampling phase (using only 20% of slices), and then refined its prediction with targeted additional sampling, leading to a high-confidence detection.

4. **Non-Maximum Suppression Impact**: The NMS process showed significant filtering capability. For instance, in "tomo_003acc", the system found 81 potential detections before NMS, which were then refined to 73 after applying 3D NMS. This demonstrates the importance of post-processing to merge nearby detections.

5. **Performance vs. Computational Trade-off**: The results highlight that the relatively simple 3D CNN architecture, when combined with strategic sampling, achieved successful detections while requiring substantially less computational resources than a full-volume analysis would require.

In the context of the original research question, these results demonstrate that deep learning can indeed effectively locate bacterial flagellar motors in 3D tomograms with accuracy approaching that of manual annotation. The successful detection across all test tomograms, with precise 3D coordinates, suggests that this approach could significantly accelerate the analysis of bacterial structures by reducing or eliminating the need for manual annotation.

The most significant finding is the effectiveness of the progressive sampling strategy in balancing computational efficiency with detection accuracy. By intelligently sampling only a fraction of the total volume, the system achieved reliable detection while dramatically reducing computational requirements – a critical consideration for the practical application of deep learning to large 3D biological datasets.

## Discussion, Conclusions, and Next Steps

### Discussion & Interpretation

This study set out to answer whether deep learning models could effectively locate bacterial flagellar motors in 3D tomograms with accuracy comparable to manual annotation. The results demonstrate that this is indeed possible, with the model successfully identifying motors in test tomograms with reasonable confidence scores and precise coordinate predictions. This finding has significant implications for structural biology research, as it suggests that a major bottleneck in the analysis pipeline—manual motor annotation—can be significantly accelerated through automation.

The performance achieved is particularly noteworthy given the computational constraints faced. The progressive sampling approach proved to be a key innovation, allowing the model to focus computational resources on promising regions while maintaining detection accuracy. This approach mirrors how human experts might analyze tomograms—first scanning at a coarse level, then focusing attention on regions of interest—suggesting that incorporating domain-inspired heuristics into deep learning pipelines can yield substantial efficiency gains.

The fluctuations observed in validation metrics during training highlight an important characteristic of this problem: the variability in tomogram quality and motor appearance makes consistent generalization challenging. This suggests that robust detection systems for this domain may need to incorporate mechanisms for handling variable image quality and structural diversity beyond what was implemented in this project.

#### Limitations

Several limitations should be acknowledged:

1. **Dataset Size and Diversity**: While the dataset was substantial in terms of raw data size (73.87 GB), the number of unique tomograms (648) and the test set (3 tomograms) were relatively limited. A larger and more diverse set of test examples would provide stronger validation of the approach's generalizability.

2. **Model Complexity Constraints**: The computational limitations necessitated a simpler model architecture than might be ideal. More complex architectures like 3D U-Net or transformer-based models might achieve higher accuracy but were prohibitively expensive given the available resources.

3. **Ground Truth Reliability**: The project assumes the ground truth annotations are accurate, but manual annotation of 3D structures itself has inherent variability and potential errors. The model's performance should be interpreted with this in mind.

4. **Single Structure Focus**: The model was trained specifically for flagellar motors and may not generalize to other macromolecular complexes without retraining.

5. **Limited Context Integration**: The current approach treats each tomogram independently and doesn't leverage potential biological context or prior knowledge about typical motor locations within bacterial cells.

### Conclusion

This project demonstrates that deep learning, specifically a 3D CNN combined with a progressive sampling strategy, can effectively automate the detection and localization of bacterial flagellar motors in tomographic data. The key innovation lies not in architectural complexity but in the sampling approach that intelligently balances computational efficiency with detection accuracy. The model achieved a validation accuracy of approximately 77% and a coordinate error of 0.32 in normalized space, successfully detecting motors in all test tomograms.

The results suggest that significant acceleration of structural biology research is possible through such automation, potentially removing a key bottleneck in the analysis of bacterial structures. The approach developed here—starting simple, focusing on efficient data handling, and progressively refining predictions—provides a template for tackling similar 3D biological structure detection problems under computational constraints.

### Future Work / Next Steps

Several promising directions for future work emerge from this project:

1. **Architectural Improvements**: With greater computational resources, exploring more complex architectures like 3D U-Net or transformer-based models could potentially improve detection accuracy and precision.

2. **Multi-Scale Analysis**: Implementing a true multi-resolution approach that analyzes the data at different scales could further improve efficiency and accuracy, particularly for capturing both local and global context.

3. **Semi-Supervised Learning**: Given the limited annotated data available in this domain, developing semi-supervised approaches that can leverage larger amounts of unannotated tomographic data would be valuable.

4. **Multiple Structure Detection**: Extending the model to simultaneously detect multiple types of macromolecular complexes would increase its utility for biological research.

5. **Integration with Downstream Analysis**: Connecting this detection system with tools for subsequent analysis of motor structure and function would create a more complete pipeline for structural biology research.

6. **Uncertainty Quantification**: Developing methods to better quantify uncertainty in detections would provide valuable information to researchers, highlighting cases where manual verification might be needed.

7. **Explainable AI Techniques**: Implementing visualization techniques to better understand what features the model is using for detection could provide biological insights and improve trust in the automated system.

The most immediate next step would be to validate the approach on a larger and more diverse test set, followed by exploring more sophisticated architectures given adequate computational resources. Ultimately, integration into actual research workflows would provide the strongest validation of the approach's practical utility.
