Abstract:

Brain tumors, characterized by the rapid growth of abnormal cells, present a significant health risk, potentially leading to severe impairment or death. Early and accurate detection is essential for effective treatment and improved patient outcomes. This project explores a deep learning-based approach for brain tumor detection using MRI images, focusing on the challenges posed by limited data availability. The approach employs the EfficientNetB0 model, selected for its balanced compound scaling, which provides high accuracy while maintaining computational efficiency. The methodology integrates transfer learning, data augmentation, and class weighting techniques to address the typical constraint of limited annotated datasets in medical imaging. This study highlights the clinical potential of deep learning models for reliable tumor detection, contributing key advancements such as the optimization of EfficientNetB0 for small medical datasets and the implementation of regularization techniques to prevent overfitting while maintaining model performance.

Aim of the Project:

This project presents a methodology for developing a model to detect brain tumors despite a small dataset. As AI advances in healthcare, it transforms diagnostics, with millions of imaging tests conducted annually, highlighting the need for efficient solutions. Computer-aided systems can enhance detection accuracy and speed over conventional methods. This study targets brain tumors, primarily detected via MRI, the gold standard in neurological imaging. Despite its effectiveness, MRI interpretation demands specialized radiologists and is time-consuming, often leading to delays and errors. Deep learning models can address these issues by offering fast, consistent analysis when trained on verified datasets.

Related Work:

A study by Prasanthi, T. Lakshmi, and Neelima, N. proposed a method for categorizing brain tumors using deep learning frameworks like DenseNet121, EfficientNet B7, InceptionResNetV2, Inception_V3, ResNet50V2, VGG16, VGG19, Xception, and a custom CNN model. The study demonstrated the effectiveness of these models and their potential to improve the speed and accuracy of brain tumor detection, enhancing patient outcomes [6].
Another study by Abdusalomov, A. B., Mukhiddinov, M., and Whangbo, T. K. (2023) proposed an approach for brain tumor detection using the YOLOv7 algorithm, enhanced with image augmentation and attention mechanisms like CBAM. The study demonstrated the effectiveness of these techniques in improving feature extraction and focusing on tumor-related regions, contributing to more accurate brain tumor detection [2].
Suitability of Different Deep Learning Approaches
For this study, various deep learning architectures for brain tumor detection were explored, including ResNet50, MobileNetV2, and EfficientNetB0. Basic CNNs, while prevalent in medical image classification, require large datasets, risking overfitting and poor generalization on small medical datasets, prompting a focus on advanced architectures.
ResNet50 excels with large datasets but poses challenges for smaller medical imaging sets. Its residual blocks with bottleneck designs demand substantial training examples to optimize skip connections over convolutional paths [7]. Moreover, its layered structure—capturing basic features early and complex abstractions later—relies on large data to minimize redundancy and noise. Its batch normalization layers further complicate training on small datasets, as limited samples yield unreliable batch statistics.
Conversely, lightweight architectures like MobileNetV2 and EfficientNetB0 suit tasks with limited data. MobileNetV2 reduces parameters via depthwise separable convolutions [4], yet its reduced capacity may limit generalization on complex MRI patterns, such as tumor boundaries, and its hyperparameter sensitivity can destabilize performance. EfficientNetB0 emerges as highly suitable due to its compound scaling strategy, balancing depth, width, and resolution. This enhances feature extraction on small datasets, ensuring stability during optimization [3]. It effectively captures tissue abnormalities and structural patterns critical for binary classification in the data, potentially leading to high accuracy, as evaluated in the Experimental Results section.

Achievements of the Project:

The project documents the methodology and experimental results of the EfficientNetB0-based model, highlighting key innovations, including data handling techniques that preserve the diagnostic features of the deep learning model, optimized transfer learning through selective layer unfreezing, and comprehensive architecture evaluation. With 95% precision for tumor detection and 91% overall accuracy, the model demonstrates clinical potential while addressing the data constraint of limited annotated medical imaging data. This approach offers a practical solution for resource-constrained medical environments where data acquisition remains challenging, while still achieving impressive results.

Project Structure:

The project structure is as follows: The Introduction reviews the aim, the Proposed Methods details the methodology, Experimental Results presents the outcomes, and the Summary provides key findings and conclusions.

Proposed Methods:

Model Architecture

The proposed method leverages transfer learning with EfficientNetB0 as the base model, chosen for its compound scaling of depth, width, and resolution, offering balanced performance and efficiency for medical image analysis. Implemented via TensorFlow’s Sequential API, the architecture maintains a linear input-output pathway, avoiding complex branching for simplicity. The base model is loaded without its top layer using ImageNet weights, and the last eight layers are unfrozen to adapt high-level features, with the number of layers evaluated in the Experimental Results section, relevant to MRI patterns while preserving lower-level feature extraction.
MRI images, classified as “yes” (tumor present/1) or “no” (tumor absent/0), are resized to 224×224 pixels and converted to RGB to match EfficientNetB0’s pretrained input requirements. Images are preprocessed using EfficientNet’s rescaling layer to normalize pixel values. The architecture is completed with GlobalAveragePooling2D to reduce spatial dimensions while retaining meaningful feature information, minimizing overfitting compared to flattening. Batch normalization is used twice, initially to stabilize the transition from pretrained layers and later after dropout for consistent gradient flow. The dense layer with 128 neurons and ReLU activation captures high-level features, balancing expressiveness and computational efficiency.
To prevent overfitting, dropout with a rate of 0.6 is applied, randomly deactivating neurons and enhancing generalization on limited medical datasets. Finally, a sigmoid output provides [0, 1] probabilities for binary classification.
(Figure 1: Model Architecture 
<img width="759" height="449" alt="Model Architecture" src="https://github.com/user-attachments/assets/7e655651-60c4-405f-92e8-41f31a453ec8" />
Addressing Class Imbalance with Augmentation:

The dataset was split into training (65%), validation (20%), and testing (15%) sets. Stratified sampling was employed to maintain class distribution across all splits, which is essential to preserve representative proportions of each class. A notable challenge in the dataset was the imbalance between the tumor and non-tumor classes. To address this, a two-pronged approach was implemented:
First, targeted data augmentation via ImageDataGenerator was applied to achieve a balanced class distribution of approximately 150 samples per class. This method calculated the exact number of synthetic samples needed for each class and generated them accordingly. The augmentation pipeline included several transformations: rotations of up to 10 degrees, width and height shifts of 10%, shear transformations at 10%, brightness variations (ranging from 0.3 to 1.0 of the original brightness), and horizontal flipping. These parameters were carefully calibrated to ensure that the augmented images remained medically plausible with their diagnostic characteristics. Secondly, class weights, computed with the ‘balanced’ setting, assign higher penalties to the minority class proportional to its inverse frequency, mitigating imbalance during loss calculation.
Model Training Strategy
The training data was processed using the ImageDataGenerator with real-time augmentation, while the validation and test sets remained constant to evaluate the model’s real-world performance. The model was trained with the Adam optimizer algorithm (learning rate = 0.0001, batch size = 16) for up to 30 epochs, utilizing callbacks. Metrics such as accuracy, loss, AUC, precision, and recall were monitored throughout training, with detailed settings and results provided in the Experimental Results section.
The model training process incorporates three key callbacks to enhance performance and prevent overfitting. Early stopping monitors validation loss and halts training when no improvement is observed over four consecutive epochs. This approach not only prevents overfitting but also restores the model’s weights from the epoch with the best validation loss, ensuring the best-performing model is retained. Additionally, learning rate reduction on plateau is applied, which reduces the learning rate by a factor of 0.2 if validation loss stagnates for four epochs. This strategy allows the model to explore a wider solution space initially and fine-tune more precisely during later stages of training.
To ensure the entire dataset is processed, the number of batch iterations per epoch is calculated as follows:
steps_per_epoch = ⌈Total augmented training set samples / Batch size⌉ = ⌈300 / 16⌉ = 19
Further, training progress was monitored with verbose output (verbose=1), tracking key performance metrics.
Experimental Results:
Hyperparameters:

The model was trained with consistent hyperparameters to ensure stability and reproducibility across experiments. A batch size of 16 was chosen to balance computational efficiency and gradient stability, given the small dataset of 223 images. Smaller batches (e.g., 8) risked producing noisy gradients with augmented data, while larger batches (e.g., 32) decreased the frequency of weight updates, potentially slowing convergence.
A learning rate of 0.0001 was set for the Adam optimizer to provide fine-grained updates suitable for transfer learning with EfficientNetB0’s pre-trained ImageNet weights. This small rate minimized the risk of drastic weight shifts in the frozen layers, preserving their generalization capabilities. The model was trained for 30 epochs—enough to learn from the augmented data while callbacks were used to prevent overfitting, as described in the Model Training Strategy section.
A Dense layer with 128 units was included to balance model complexity and stability while penalizing large weights to prevent overfitting—a concern given the dataset’s limited size. After testing dropout rates of 0.3, 0.6, and 0.7, a rate of 0.6 was selected. The rate of 0.3 resulted in significant overfitting, while 0.7 led to slight underfitting due to excessive regularization. Consequently, dropout of 0.6 provided the best balance.
The decision to unfreeze the last eight layers of EfficientNetB0 was validated through experiments with 3, 5, and 8 unfrozen layers. Unfreezing 8 layers provided the optimal balance between retaining ImageNet’s foundational feature extraction capabilities while allowing sufficient adaptation to tumor-specific patterns. Unfreezing three layers limited model expressiveness, reducing performance due to insufficient fine-tuning of tumor-relevant features, while unfreezing five layers showed improvement but didn’t fully capture the complex patterns present in brain MRI data. By unfreezing eight layers, the model gained enough flexibility to learn domain-specific features while still benefiting from the pre-trained weights of the remaining layers.


Metrics Analysis:
<img width="571" height="252" alt="Metrics Analysis" src="https://github.com/user-attachments/assets/e5ac54a4-6570-4bff-becc-94e4c902e0b5" />


ClassPrecisionRecallF1-ScoreSupportNo Tumor0.830.910.8711Tumor0.950.910.9323Accuracy0.9134Macro avg0.890.910.9034Weighted avg0.920.910.9134
The model achieved exceptional performance metrics on the test set, with an overall accuracy of 91.18% and an impressive AUC of 98.4%, demonstrating excellent discriminative ability between tumor and non-tumor cases. Class-specific analysis revealed high precision (0.95) and recall (0.91) for tumor detection, yielding an F1-score of 0.93. The no-tumor class showed good performance with precision of 0.83, recall of 0.91, and F1-score of 0.87. The slightly higher precision for tumor detection is clinically significant [5], reducing unnecessary concern from false positives.
Confusion matrix analysis provided further insight into model performance: the confusion matrix was run on the test set, which comprised 15% of the data sample.

21 true positives (correctly identified tumors),
10 true negatives (correctly identified healthy brains),
1 false positive (healthy brain incorrectly classified as tumor),
2 false negatives (missed tumors).

These results yield a sensitivity of 91.3% and a specificity of 90.9%, which is impressive given our constraints. However, the presence of false negatives raises concerns, as missing actual tumors can be critical in medical applications.
Figure 2: Confusion Matrix 
<img width="365" height="335" alt="Confusion Matrix" src="https://github.com/user-attachments/assets/ef67a193-1277-457d-bc1e-839401c5e6d8" />
Figure 3: AUC Plot & ROC Plot 
<img width="747" height="388" alt="Results" src="https://github.com/user-attachments/assets/3e3e5209-3250-4296-8c2a-7549262d0cdb" />
Figure 5: Accuracy Plot & True vs Predicted Labels 
<img width="732" height="382" alt="Peformances" src="https://github.com/user-attachments/assets/8f0be451-e002-43f5-86aa-a67dae74266e" />
Figure 7: Model Predictions 
<img width="766" height="224" alt="Modell Predictions" src="https://github.com/user-attachments/assets/c586bad5-fdc0-48cb-85c2-e5be3f1d0b71" />

The ROC curve analysis confirms the model’s robust performance, with the curve sharply maintaining an AUC of 0.98.
Training dynamics analysis shows proper convergence without significant overfitting. The training accuracy progressively improved from approximately 51% to 90%, while validation accuracy quickly reached 80% before gradually improving to 87%. The consistent improvement in validation metrics throughout training indicates good generalization capability. The initial gap between training and validation performance narrowed over time, suggesting that the regularization techniques effectively prevented overfitting despite the limited dataset size.

Prediction probability analysis reveals that most samples were classified with high confidence, with few cases falling near the decision threshold. This suggests the model learned robust discriminative features, further validating its potential clinical utility for brain tumor detection.

Summary

This study demonstrates the efficacy of transfer learning with EfficientNetB0 for brain tumor detection in MRI images, achieving 91.18% accuracy despite dataset limitations. Strategic implementation of data augmentation, coupled with careful hyperparameter optimization, proved crucial for model performance. Unfreezing the last eight layers of EfficientNetB0 struck an optimal balance between leveraging pre-trained features and domain adaptation. The model’s high precision for tumor detection (95%) minimizes false positives, though the presence of two false negatives requires attention in future work.
Although implementation could benefit from added advanced segmentation techniques, exploring ensemble approaches to reduce false negatives and generalise better. This work contributes a practical approach to brain tumor detection that balances performance with implementation feasibility in resource-constrained clinical environments.


References


[1] [Online; accessed 20. Mar. 2025]. May 2023. URL: https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2023/05/Statistical-Release-18th-May-2023-PDF-471KB-1.pdf.
[2] Akmalbek Bobomirzaevich Abdusalomov, Mukhriddinov, M., and Whangbo, T. K. “Brain Tumor Detection Based on Deep Learning Approaches and Magnetic Resonance Imaging”. In: Cancers 15.16 (Aug. 2023), p. 4172. DOI: 10.3390/cancers15164172.
[3] Serra Aksoy and Pritika Dasgupta. “AI-Powered Neuro-Oncology: EfficientNetB0’s Role in Tumor Differentiation”. In: Clin. Transl. Neurosci. 9.1 (Jan. 2025), p. 2. ISSN: 2514-183X. DOI: 10.3390/ctn9010002.
[4] Andrew G. Howard et al. “MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications”. In: arXiv (Apr. 2017). DOI: 10.48550/arXiv.1704.04861. eprint: 1704.04861.
[5] Keylabs. “Techniques to Enhance Precision in Machine Learning Models | Keylabs”. In: Keylabs: latest news and updates (Sept. 2024). URL: https://keylabs.ai/blog/techniques-to-enhance-precision-in-machine-learning-models.
[6] T. Lakshmi Prasanthi and N. Neelima. “Improvement of Brain Tumor Categorization using Deep Learning: A Comprehensive Investigation and Comparative Analysis”. In: Procedia Comput. Sci. 233 (Jan. 2024), pp. 703–712. ISSN: 1877-0509. DOI: 10.1016/j.procs.2024.03.259.
[7] Resnet 18 Vs Resnet 50 Comparison | Restackio. [Online; accessed 22. Mar. 2025]. Mar. 2025. URL: https://www.restack.io/p/resnet-fine-tuning-answer-resnet-18-vs-resnet-50-cat-ai.
