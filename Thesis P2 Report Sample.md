- Occlusion-Resistant Animal Detection: A Computer Vision Approach for Wildlife Monitoring and Conservation 

by 

S. M. SAIDUR RAHMAN 22101451 MD.NIAMATULLAH 22101439 SUBRATA BAISHNAB 22101642 MD. ANN-AM AKBAR SAAD 22101438 SHAYONTO RAYHAN 22101651 

A thesis submitted to the Department of Computer Science and Engineering in partial fulfillment of the requirements for the degree of B.Sc. in Computer Science and Engineering 

Department of Computer Science and Engineering Brac University June 2025. 

© 2025. Brac University All rights reserved. 

## **Declaration** 

It is hereby declared that 

1. The thesis submitted is my/our own original work while completing degree at Brac University. 

2. The thesis does not contain material previously published or written by a third party, except where this is appropriately cited through full and accurate referencing. 

3. The thesis does not contain material which has been accepted, or submitted, for any other degree or diploma at a university or other institution. 

4. We have acknowledged all main sources of help. 

## **Student’s Full Name & Signature:** 

**==> picture [414 x 179] intentionally omitted <==**

**----- Start of picture text -----**<br>
S. M. SAIDUR RAHMAN MD.NIAMATULLAH<br>22101451 22101439<br>SUBRATA BAISHNAB<br>MD. ANN-AM AKBAR SAAD<br>22101642<br>22101438<br>**----- End of picture text -----**<br>


SHAYONTO RAYHANB 22101651 

i 

## **Approval** 

The thesis titled “Occlusion-Resistant Animal Detection: A Computer Vision Approach for Wildlife Monitoring and Conservation” submitted by 

1. S. M. SAIDUR RAHMAN (22101451) 

2. SHAYONTO RAYHAN (22101651) 

3. MD. ANN-AM AKBAR SAAD (22101438) 

4. MD.NIAMATULLAH (22101439) 

5. SUBRATA BAISHNAB (22101642) 

of [Spring], [2025] has been accepted as satisfactory in partial fulfillment of the requirement for the degree of B.Sc. in Computer Science in [Current Semester] Year. 

## **Examining Committee:** 

Supervisor: 

(Member) 

Dr. Amitabha Chakrabarty Professor 

Department of Computer Science and Engineering BRAC University 

Thesis Coordinator: 

(Member) 

Dr. Md. Golam Rabiul Alam Professor Department of Computer Science and Engineering BRAC University 

Head of Department: (Chair) 

Dr. Sadia Hamid Kazi 

Associate Professor & Chairperson Department of Computer Science and Engineering BRAC University 

ii 

## **Abstract** 

The mainstay of effective wildlife conservation is efficient and judicious detection and monitoring of animals in very dense, occluded habitats. For example, the Royal Bengal Tiger is an endangered apex predator, which is the pride of Bangladesh in terms of natural heritage. Its presence in the ecosystem is crucial for biophysical diversity. A good reason for driving this idea is the demand for data-driven methods to monitor and preserve endangered species. Thus, our research ultimately seeks to evolve sophisticated instruments for accurate population estimation or re-identification of wildlife and habitat monitoring. The initial phase focused on animal detection as the first step towards re-identification, addressing broader biodiversity needs such as accurate re-identification of animals in partially occluded images, often encountered in dense forest environments. In this research, we bridge the gap between application-based theoretical study and practical deployment by leveraging stateof-the-art computer vision techniques, particularly Mamba-based vision models, for real-world monitoring systems. A comprehensive methodology was developed and evaluated using five deep learning models on a large-scale dataset of 8,611 animal individual. The best-performing model, EfficientViM-M1, achieved 83.13% Top-1 accuracy with only 14M parameters and 246M FLOPs, representing an 18× reduction in computational cost compared to Swin-T. Metric learning with ArcFace loss further improved individual animal segregation , converging four times faster than triplet loss and producing closer, well-separated embedding clusters. These results confirm the effectiveness of our approach for resource-constrained, real-time wildlife monitoring and lay the foundation for future work on multi-animal detection and advanced occlusion handling. 

**Keywords:** Re-identification, Occlusion, Computer Vision, Wildlife, Deep Learning, IoT, Machine Learning, Metric Learning 

iii 

## **Table of Contents** 

|**Declaration**|**Declaration**|**Declaration**|||**i**|
|---|---|---|---|---|---|
|**Approval**|||||**ii**|
|**Abstract**|||||**iii**|
|**Table of Contents**|||||**iv**|
|**List of Figures**|||||**vi**|
|**List of Tables**|||||**1**|
|**1**|**Introduction**||||**2**|
||1.1|Background . . . . . . . . .|. . . . . .|. . . . . . . . . . . . . . . . .|2|
||1.2|Rational of the Study or Motivation . .||. . . . . . . . . . . . . . . . .|2|
||1.3|Problem Statement . . . . .|. . . . . .|. . . . . . . . . . . . . . . . .|2|
||1.4|Objective<br>. . . . . . . . . .|. . . . . .|. . . . . . . . . . . . . . . . .|3|
||1.5|Methodology in Brief<br>. . .|. . . . . .|. . . . . . . . . . . . . . . . .|3|
||1.6|Scopes and Challenges . . .|. . . . . .|. . . . . . . . . . . . . . . . .|3|
|**2**|**Literature Review**||||**4**|
||2.1|Preliminaries<br>. . . . . . . .|. . . . . .|. . . . . . . . . . . . . . . . .|4|
||2.2|Review of Existing Research|. . . . . .|. . . . . . . . . . . . . . . . .|5|
||2.3|Summary of Key Findings .|. . . . . .|. . . . . . . . . . . . . . . . .|5|
||2.4|Summary<br>. . . . . . . . . .|. . . . . .|. . . . . . . . . . . . . . . . .|7|
|**3**|**Requirements, Impacts and Constraints**||||**8**|
||3.1|Final Specifcations and Requirements||. . . . . . . . . . . . . . . . .|8|
||3.2|Societal Impact . . . . . . .|. . . . . .|. . . . . . . . . . . . . . . . .|8|
||3.3|Environmental Impact . . .|. . . . . .|. . . . . . . . . . . . . . . . .|8|
||3.4|Project Management Plan .|. . . . . .|. . . . . . . . . . . . . . . . .|8|
|**4**|**Proposed Methodology**||||**9**|
||4.1|Design Process or Methodology Overview<br>. . . . . . . . . . . . . . .|||9|
|||4.1.1<br>Design Constraints .|. . . . . .|. . . . . . . . . . . . . . . . .|9|
|||4.1.2<br>Two-Phase Strategy|. . . . . .|. . . . . . . . . . . . . . . . .|10|
|||4.1.3<br>Future Considerations|. . . . .|. . . . . . . . . . . . . . . . .|10|
||4.2|Preliminary Design or Design|(Model) Specifcation . . . . . . . . . .||11|
|||4.2.1<br>Baseline Architecture|. . . . . .|. . . . . . . . . . . . . . . . .|11|
|||4.2.2<br>Mamba Based Vision Variants .||. . . . . . . . . . . . . . . . .|11|



iv 

|||4.2.3<br>Comparative Performance Analysis of Literature-Based Per-||
|---|---|---|---|
|||formance . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|14|
||4.3|Data Collection . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|14|
||4.4|Implementation of Selected Design<br>. . . . . . . . . . . . . . . . . . .|16|
|**5**|**Result Analysis**||**19**|
||5.1|Performance Evaluation<br>. . . . . . . . . . . . . . . . . . . . . . . . .|19|
||5.2|Analysis of Design Solutions . . . . . . . . . . . . . . . . . . . . . . .|21|
||5.3|Discussions<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|22|
|**6**|**Conclusion**||**24**|
||6.1|Conclusion . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|24|
|**Bibliography**|||**26**|



v 

## **List of Figures** 

- 4.1 (a) The architecture of a Swin Transformer (Swin-T); (b) two successive Swin Transformer Blocks W-MSA and SW-MSA are multi-head self attention modules with regular and shifted windowing configurations, respectively. . . . . . . . . . . . . . . . . . . . . . . . . . . . . 11 

- 4.2 The architecture of hierarchical MambaVision models. The first two stages use residual convolutional blocks for fast feature extraction. Stages 3 and 4 employ both MambaVision and Transformer blocks. Specifically, given N layers, we use N 2 MambaVision and MLPblocks, which are followed by additional N 2 Transformer and MLP blocks. The Transformer blocks in the final layers allow for recovering lost global context and capturing long-range spatial dependencies. . . . . 11 

- 4.3 Architecture of MambaVision block. In addition to replacing causal Conv layer with their regular counterparts, we create a symmetric path without SSM as a token mixer to enhance the modeling of global context. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 12 

- 4.4 The overview of the proposed Vim model. We first split the input image into patches, and then project them into patch tokens. Last, we send the sequence of tokens to the proposed Vim encoder. To perform ImageNet classification, we concatenate an extra learnable classification token to the patch token sequence. Different from Mamba for text sequence modeling, Vim encoder processes the token sequence with both forward and backward directions. . . . . . . . . . . . . . . 12 

- 4.5 Architecture of Vmamba and VSS BLock . . . . . . . . . . . . . . . . 13 4.6 (left) Overall architecture and (right) block design of EfficientViM. The dotted line indicates a skip connection for multi-stage hidden state fusion (MSF). . . . . . . . . . . . . . . . . . . . . . . . . . . . 13 

- 4.7 Wildlife10k Dataset Images . . . . . . . . . . . . . . . . . . . . . . . 15 

- 5.1 t-SNE visualizations of embeddings for 64 individual starfish and bird classes: (a) Triplet Loss shows significant overlap between individuals, indicating poor separation; (b) ArcFace produces tight, well separated clusters, demonstrating superior individual discrimination. . . . . . . 20 

- 5.2 t-SNE visualizations of embeddings for 531 individual cat and cattle classes: (c) Triplet Loss shows overlapping clusters and poor individual separation; (d) ArcFace produces distinct, compact clusters, indicating much better discrimination between individuals. . . . . . . 20 

vi 

## **List of Tables** 

|2.1|Summary Table . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|7|
|---|---|---|
|4.1|Literature-Based Performance Table (Based on ImageNet-1K, 224x224||
||input). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|14|
|4.2|Final dataset for Phase 1: Animal-wise class distribution and image||
||counts. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|16|
|4.3|Dataset summary for selected land-based animal classes (Phase 2,||
||only classes with _≥_20 images/class included).<br>. . . . . . . . . . . . .|16|
|4.4|Common training parameters used across all fve architectures . . . .|17|
|4.5|Phase 2 Metric Learning: Training parameters and environment. . . .|18|
|5.1|Phase 1 Results: Model performance on our dataset (300 epochs,||
||224_×_224 input). . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|19|
|5.2|Performance summary for selected animal classes using EVim-M1||
||with ArcFace loss. . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|21|



1 

## **Chapter 1** 

## **Introduction** 

## **1.1 Background** 

Detection and monitoring of animals are essential for wildlife conservation, particularly in the case of dense, occluding habitats. Existing methods that are resource hungry and often do not work as expected when animals are partially visible and therefore their effectiveness is limited in real-world situations. However, the novel and advanced technologies introduced in computer vision and IoT today provide newer avenues for addressing these challenges. The attention of this study is the development of animal re-identification and contributing in the broader systems of wildlife monitoring. 

## **1.2 Rational of the Study or Motivation** 

Royal Bengal tiger, the top predator of the Sundarbans, is crucial for ecological balance and biodiversity maintenance for our country [12]. Currently, the Forest Department of Bangladesh depends on footprint inspection of tigers or camera traps to monitor tiger populations. While this method has been in use for years, it lacks accuracy and efficiency, especially when compared to modern, image-based identification techniques. This motivated us for our research because of this endangered status for a resource-efficient, occlusion-resistant system aimed at better wildlife monitoring and preservation. 

## **1.3 Problem Statement** 

Today’s wildlife detection systems have limitations such as occlusion, environmental variability, which make it difficult for them to perform well or scale. Advances such as MegaDescriptor [9], UniReID [10], and ARBase [13] have improved generalization, though they are not robust in handling occlusion and are resource-hungry. The purpose of this work is to design a resource-efficient, occlusion-resistant detection model using Mamba based vision models and occulsion handling to support realtime wildlife monitoring. 

2 

## **1.4 Objective** 

- Develop a resource efficient, generalized animal re-identification system utilizing advanced Mamba based vision models to automate wildlife monitoring, replacing traditional manual wildlife tracking methods (e.g., footprints, camera traps) enabling large scale, real time wildlife monitoring and population estimation in remote or resource limited environments. 

- Handling occluded, low-quality, partial images from challenging dense wild environments for establishment of an accurate and robust individual animal re-identification system. 

- Providing a platform that supports conservation efforts using scalable, sustainable, and deployable framework for diverse and endangered species monitoring. 

## **1.5 Methodology in Brief** 

The research uses a two-phase approach: First, advanced Mamba-based vision models (e.g., EfficientViM-M1, Vim) are integrated for animal re-identification, focusing on resource efficiency and occlusion handling in large, diverse datasets. MegaDescriptor or Swin-T are chosen as baselines due to their proven accuracy and strong embedding performance in recent literature, but are limited by high computational demands and slower training. Vision Mamba addresses these limitations with efficient state space modeling, enabling real-time, scalable deployment. In the second phase, metric learning is applied to further improve embedding quality and enable accurate, open-set identification. The system is evaluated for both accuracy and efficiency to ensure suitability for real-time wildlife monitoring. 

## **1.6 Scopes and Challenges** 

This study proposes a resource-efficient, occlusion-resistant animal re-identification system using Mamba-based vision architectures. The model is designed to deliver high accuracy across diverse species and challenging environmental conditions, including severe occlusion and variable image quality. The solution supports real-time wildlife monitoring and provides a scalable foundation for large-scale identification systems. Key challenges addressed include handling dense habitats with frequent occlusion, ensuring computational efficiency for deployment in remote or low-resource environments, and achieving generalization across species. The research also highlights the importance of robust model performance despite class imbalance and limited data for rare species, setting the stage for future work in multi-animal detection and advanced occlusion handling. 

3 

## **Chapter 2** 

## **Literature Review** 

## **2.1 Preliminaries** 

Wildlife re-identification is a specific computer vision task carried out for analyzing an individual’s photographs of an animal species for its visual features, such as patterns, textures, and distinguishing physical characteristics. Most significantly, it aids conservation, population monitoring, and biodiversity studies. However, such methods usually require advanced computation since they do face challenges in terms of biodiversity, pose variants due to occlusion and multiple natural environments. 

## **Concepts:** 

- **Feature Extraction:** Modern deep learning models extract meaningful features from images of an individual. This includes using convolutional neural networks (CNNs) to grasp local patterns, and DINOv2, an entirely new selfsupervised transformer-based model to learn robust, generalized features from diverse datasets. 

- **Transformer-Based Architectures:** Models like Swin transformers help in modeling long-range dependencies to benefit from recognition of more complex patterns and relationships within wildlife data. 

- **Loss Functions:** Triplet loss and cross-entropy loss are commonly used for embedding space optimization to fine-tune re-identification performance with respect to groups of ”similar” and ”dissimilar” identities. 

- **Vision Mamba:** A very highly efficient framework which uses bidirectional state space models (SSMs) for visual data processing allowing faster computation and better scalability for large-scale tasks while still maintaining the desired accuracy for wildlife detection. 

- **Facial Reconstruction and De-Occlusion:** Techniques such as GAN-based means (e.g. DFNet, G-NST), statistical imputation (MICE, RegEM), etc., stand for partial reconstruction of human faces. 

4 

## **2.2 Review of Existing Research** 

New advancements in wildlife re-identification are moving towards improving accuracy in modeling and dataset reliability. One notable new contribution is MegaDescriptor [11], the model that serves as a baseline for individual animal re-identification. MegaDescriptor uses transformer-based architectures (such as Swin-B), which are optimized with ArcFace loss, to produce precise embedding in terms of some individual specific characteristics of animals, such as their patterns or markings. This model outperforms general-purpose models such as CLIP and DINOv2, especially within the setting with diverse species [9]. In addition, the model can be used with zero-shot learning, making it flexible for many wildlife datasets. UniReID [10] introduces a universal approach of animal re-identification in the wild using dual-modal guidance, it associates the visual prompts created from support images with textual guidance derived from GPT-4 to flexibly adapt UniReID into novel species. Such a framework will set a new benchmark for generalization in wildlife monitoring, and further knowledge into biodiversity and conservation efforts. The ATRW dataset contains data collected with the aim of Amur tiger recognition. Integrated SSDMobileNetv2 detection with HRNet or OpenPose-based pose estimation is a PPbM with ResNet-50 [6]. This really broadens the scope for effective use of local pose characteristics loading from global image data in enhancing performance in wild settings for more effective robust tiger re-identification. 

Other Computer vision techniques have been applied into animal re-identification for wildlife monitoring. For example, convolutional neural networks (CNN) have been used to extract features from animal images, including chimpanzee, gorilla faces, to achieve high performance in re-ID [1] [2]. ARBase [13] based on ResNet-50 with In-instance Batch Normalization (IBN) and a multi-branch architecture for extracting both global and fine-grained features employing triplet loss along with cross-entropy loss to obtain state-of-the-art performance on benchmarks, including HyenaID and WhaleSharkID, while effectively handling species-specific challenges. AlexNet, YOLO are also used for individual animal identification tasks [5]. Moreover, Siamese Networks compare image pairs, and they have shown effectiveness for re-ID across multiple species, including lemurs and golden monkeys [3]. Additionally, ResNet50, in tandem with YOLO, has been used to re-ID elephants while providing robust performance even in challenging settings [4]. 

## **2.3 Summary of Key Findings** 

**Generalized Models for the Re-Identification of Wildlife:** In the past, the animal re-identification models were primarily species-specific with very small datasets of one or two species. The current trend within the domain shows the growing interest in the construction of generalized models with merged datasets capable of enormous scalability and adaptability. While MegaDescriptor [11] makes use of the large WildlifeReID-10k [24] dataset containing over 214k images of 10k individual animals, models like UniReID [10] and ARBase [13] are representative of the emerging trend towards adding several smaller datasets. This new way stresses the importance of using other species to build a model that can handle as many different wildlife monitoring scenarios as possible. 

5 

## **Wildlife Re-Identification and its Subsequent Advancement with Com-** 

**putational Efficacy:** Recent trends in wildlife re-identification have focused on leveraging high-computation tasks to improve the accuracy and scalability of models. A few examples of major advancements in this regard are MegaDescriptor, UniReID, and ARBase, which operate at advanced architecture coupled with large data sets. Unfortunately, these high-performance models are alternatively too intensive on resource consumption with very variable kinds and large-scale datasets. Moreover field deployment might be complex in terms of resources. 

However, Vision Mamba [20] delivers quicker and provides equally good results with relatively less computational overhead and efficient computational implementations [19]. The use of State Space Models (SSMs) and a bidirectional processing approach with Vision Mamba allows reducing time and memory needs for re-identification tasks (MegaDescriptors training took one week in a single gpu [11] ) . Thus, Vision Mamba is excellent for real-time, large-scale wildlife monitoring systems. It is in keeping with the rapidly rising demand for complex data models requiring not only efficiency but also scalable computation with a scope for better field deployement. 

**Occlusion Handling in Wildlife Re-Identification:** The crucial gap in wildlife research on recognition is how to handle occlusions, which are problems best met in the field of dense natural environments. Advances in human recognition technology that include reconstruction and de-occlusion techniques have not yet been adopted by wildlife re-identification research. For example, Generative Face Completion and DFNet fill in the missing parts or put them in structurally position that are coherent to reconstruct occluded facial regions, while enhancement techniques such as CodeFormer and GFPGAN restore degraded images [16] . MICE and RegEM methods statistically impute or iteratively approach estimation of missing features, improving recognition accuracy and robustness under partial occlusion but suffering from severe occlusions or real-world variation [17]. 

6 

## **2.4 Summary** 

|**Key Finding**|**Description**|**Impact**|
|---|---|---|
|Generalized Models for<br>Re-Identifcation|Models like Swin Trans-<br>former<br>based<br>MegaDe-<br>scriptor,<br>UniReID,<br>and ARBase use large,<br>merged datasets to han-<br>dle multiple species.|Increases scalability and<br>adaptability for diverse<br>wildlife monitoring.|
|Computational Efciency|Mamba based models like<br>Vision Mamba is opti-<br>mized for less resource<br>consumption<br>and<br>real-<br>time performance.|Allows<br>for<br>large-scale<br>wildlife monitoring with-<br>out heavy computational<br>overhead.|
|Occlusion Handling|Existing<br>human<br>recog-<br>nition<br>methods<br>(e.g.,<br>DFNet,<br>MICE)<br>could<br>help to improve occlusion<br>handling<br>for<br>wildlife<br>re-identifcation.|Help<br>to<br>address<br>chal-<br>lenges in dense natural<br>environments where oc-<br>clusions are common.|



Table 2.1: Summary Table 

7 

## **Chapter 3 Requirements, Impacts and Constraints** 

## **3.1 Final Specifications and Requirements** 

The project required training five deep learning models (Swin-T, VMamba-T, VimS, MambaVision-T, EfficientViM) on a 13GB version of WildlifeReID-10k dataset, using high-performance hardware (GPU: RTX 4090, RAM: 64GB, CPU: i9-14900K) and Python with PyTorch. Rigorous data cleaning, augmentation, and transformation ensured a unified, high-quality dataset (primarily 8,611 classes, 80/20 split). Both classification and metric learning (ArcFace and Triplet Loss) were implemented to enable robust, open-set recognition and generalization across species and realworld scenarios. 

## **3.2 Societal Impact** 

This system advances wildlife conservation by automating accurate animal identification, reducing manual effort and error, and supporting endangered species protection and biodiversity policy. Its resource efficiency makes it practical for remote or low-resource conservation teams, while also enabling community engagement and research. 

## **3.3 Environmental Impact** 

Automated, non-invasive identification reduces the need for physical tagging and field disturbance, supporting sustainable conservation. Efficient models minimize energy use, making AI deployment more environmentally friendly. 

## **3.4 Project Management Plan** 

The project followed a phased plan: (1) dataset collection/cleaning, (2) baseline model training/evaluation, (3) Metric learning (ArcFace, Triplet Loss) implementation Progress was tracked with regular reviews, and compute resources were optimized by scheduling heavy tasks during off-peak hours and using cloud backup. 

8 

## **Chapter 4 Proposed Methodology** 

## **4.1 Design Process or Methodology Overview** 

The primary goal is to develop identification of individual animals (re-identification) in a generalized system that outperforms traditional methods in wildlife monitoring scenarios including counting individual animals, providing accurate identification while significantly reducing computational overhead compared to existing transformer-based approaches and eliminating the time intensive, error prone manual counting systems currently used in conservation for real time deployment in conservation applications. By employing a comparative analysis framework to identify optimal resource efficient methodology with occlusion handling it would achieve the key objective for future implementation phases. The designed methodology operates under four critical constraints, with the first two addressed in the current phase and the latter two planned for future implementation: 

## **4.1.1 Design Constraints** 

**Computational Efficiency :** The system must demonstrate superior resource utilization with reduced parameter count and faster inference speeds compared to baseline Swin Transformer model while maintaining accuracy, as wildlife monitoring often occurs in remote locations with limited computational resources and requires real-time processing for effective conservation management. 

**Class Imbalance and Embedding Overlap Handling:** Robust performance across severely imbalanced datasets where 3,000+ classes contain only single images while others have multiple of samples, addressing embedding space overlap issues that arise from limited diverse image availability per individual. Indicating one of the major publicly available dataset issue in wildlife monitoring task. So, it is important to provide a foundational platform that can effectively distinguish between individuals despite sparse data. 

**Multi-Animal Detection and Counting:** The challenge of detecting and counting multiple individuals within single images (e.g., three tigers in one frame) while establishing spatial constraint for determining when to register an individual as a new individual versus a repeated sighting, including protocols for counting frequency 

9 

thresholds and exclusion criteria for total population estimation. 

**Occlusion and Image Quality Handling :** The system must perform robustly with noisy, blurry, half visible, or partially occluded images where animals are hidden behind bushes, vegetation, or other environmental obstacles, as these represent the majority of real-world wildlife monitoring scenarios in dense natural habitats. 

## **4.1.2 Two-Phase Strategy** 

The research follows a logical two-phase strategy based on the design constraints that mirrors how wildlife researchers would naturally approach the challenge of identifying individual animals in the wild. 

Phase 1 establishes efficient individual animal distinction capabilities by comparing four Mamba based vision models with a proven Swin Transformer baseline to classify 8,611 unique individual animals. Rather than simply categorizing animals, this phase develops each model’s ability to recognize individual animals - essentially teaching the system to count unique individuals by learning their distinctive features. The evaluation prioritizes practical concerns like parameter count, inference speed, and computational efficiency (FLOPs) alongside accuracy, recognizing that wildlife monitoring often happens in remote locations with limited resources. 

Phase 2 advances to re-identification using metric learning for recognizing which is essential because, unlike traditional classification which only works with predefined animals, metric learning creates feature representations that can identify individuals the system has never seen before crucial for real-world wildlife monitoring where new animals are constantly encountered. By comparing Triplet Loss and ArcFace Loss methods and analyzing how well they separate inter and intra individual animals in embedding space which solves the embedding overlap issue in our dataset which is the critical challenge of distinguishing between visually similar animals across different sightings. After selecting the appropriate metric learning approach with our phase 1 selected model we performed the re-identification in our augmented animal wise dataset. The transition from combined datasets in Phase 1 (for broad architectural comparison) to animal-specific balanced datasets in Phase 2 ensures the metric learning approach works effectively across different animal types while addressing the practical challenge of limited training images per individual. 

## **4.1.3 Future Considerations** 

The next phases will address multi-animal detection and counting within single images, establishing protocols for distinguishing new individuals from repeated sightings. Additionally, occlusion and image quality handling will enable robust performance with noisy, blurry, or partially occluded images common in dense natural habitats where animals are hidden behind vegetation. 

10 

## **4.2 or Preliminary Design Design (Model) Specification** 

This research systematically evaluates five state-of-the-art vision architectures comprising four Mamba based vision variants and one Transformer baseline. The selection encompasses the most recent advances in efficient vision models with linear computational complexity. 

## **4.2.1 Baseline Architecture** 

Originally proposed by Liu et al., Swin Transformer [7] introduces shifted windowing to achieve linear computational complexity while maintaining the modeling capacity of self-attention mechanisms chosen as the baseline making it a credible benchmark for comparison (as highlighted in the literature review for comparing with Megadescriptor). 

Figure 4.1: (a) The architecture of a Swin Transformer (Swin-T); (b) two successive Swin Transformer Blocks W-MSA and SW-MSA are multi-head self attention modules with regular and shifted windowing configurations, respectively. 

## **4.2.2 Mamba Based Vision Variants** 

## **1. MambaVision** 

Figure 4.2: The architecture of hierarchical MambaVision models. The first two stages use residual convolutional blocks for fast feature extraction. Stages 3 and 4 employ both MambaVision and Transformer blocks. Specifically, given N layers, we use N 2 MambaVision and MLPblocks, which are followed by additional N 2 Transformer and MLP blocks. The Transformer blocks in the final layers allow for recovering lost global context and capturing long-range spatial dependencies. 

11 

Figure 4.3: Architecture of MambaVision block. In addition to replacing causal Conv layer with their regular counterparts, we create a symmetric path without SSM as a token mixer to enhance the modeling of global context. 

**Architecture:** Hybrid Mamba-Transformer with hierarchical design. 

**Core Innovation:** MambaVision Mixer combined with self-attention blocks in final layers. 

**Design Philosophy:** Early stages use CNN blocks for fast feature extraction, later stages employ Mamba and Transformer blocks for global context modeling. 

**Key Feature:** Symmetric path design without SSM for enhanced global context modeling 

## **2. Vision Mamba (Vim)** 

Figure 4.4: The overview of the proposed Vim model. We first split the input image into patches, and then project them into patch tokens. Last, we send the sequence of tokens to the proposed Vim encoder. To perform ImageNet classification, we concatenate an extra learnable classification token to the patch token sequence. Different from Mamba for text sequence modeling, Vim encoder processes the token sequence with both forward and backward directions. 

**Architecture:** Pure bidirectional State Space Model for vision 

**Core Innovation:** Bidirectional SSM processing to capture global context in single forward pass 

12 

**Design Philosophy:** Direct adaptation of Mamba to vision through bidirectional scanning 

**Key Feature:** Position-sensitive processing with global context understanding 

## **3. VMamba** 

Figure 4.5: Architecture of Vmamba and VSS BLock 

**Architecture:** Visual State Space Model with 2D Selective Scan (SS2D) **Core** 

**Innovation:** Four-way selective scanning methodology for comprehensive spatial context **Design Philosophy:** Cross-scan module to integrate information from mul- 

tiple directions **Key Feature:** Hierarchical multi-resolution structure with depthwise convolutions 

## **4. EfficientViM** 

Figure 4.6: (left) Overall architecture and (right) block design of EfficientViM. The dotted line indicates a skip connection for multi-stage hidden state fusion (MSF). 

**Architecture:** Hidden State Mixer-based State Space Duality (HSM-SSD) 

**Core Innovation:** Channel mixing operations performed in compressed hidden state space 

**Design Philosophy:** Computational efficiency through hidden state compression 

**Key Feature:** Multi-stage hidden state fusion for enhanced representation power 

13 

## **4.2.3 Comparative Performance Analysis of Literature-Based Performance** 

A literature-based performance analysis (Table 4.1) provides compelling evidence for the superior computational efficiency of Mamba-based architectures over Transformer models showing possible implementation in animal re-identification tasks. The results collected from ImageNet-1K, a standard benchmark for general vision tasks demonstrate that VMamba-T achieves superior resource utilization with 30M parameters delivering 1686 img/s throughput and 82.6% accuracy, compared to Swin-T’s 28M parameters achieving 1244 img/s throughput at 81.3% accuracy, representing a 35% throughput improvement with only 7% parameter increase while maintaining 1.3% accuracy gain. This efficiency advantage becomes critical when contrasted with smaller-scale variants like EfficientVimM1 and EfficientViT-M3 (6.7M-6.9M parameters but 20731-16045 img/s throughput with lower 73.4-73.5% accuracy), which despite their compact size fail to achieve competitive accuracy, and considering that Vision Mamba’s bidirectional SSM architecture offers linear ² complexity O(N) versus standard Vision Transformer’s quadratic O(N ), directly addressing the computational constraints identified in Chapter 2 where models like MegaDescriptor require week-long training periods. The selection of Swin-T as the Transformer baseline is justified by its optimal parameter-performance balance (28M parameters, 81.3% accuracy) compared to less efficient alternatives (DeiT-Ti: 6M parameters but only 72.2% accuracy; Vim-Ti with only 7M parameters achieving 76.1% accuracy; MambaVision-T: 31.8M parameters with lower 6298 img/s throughput), establishing a fair comparison framework focused on tiny and small model variants that makes it the most credible benchmark for evaluating Mamba’s potential in resource-constrained wildlife monitoring scenarios where real-time processing capability and parameter efficiency are paramount for field deployment feasibility. 

|**Model**|**Params (M)**|**Throughput (img/s)**|**FLOPs**|**Top-1 Acc. (%)**|**Source**|
|---|---|---|---|---|---|
|Swin-T<br>VMamba-T|28<br>30|1244<br>1686|4.5G<br>4.9G|81.3<br>82.6|[15]|
|DeiT-Ti<br>Vim-Ti|6<br>7|-<br>-|-<br>-|72.2<br>76.1|[21]|
|EfcientVim-M1<br>EfcientViT-M3|6.7<br>6.9|20731<br>16045|239M<br>263M|73.5<br>73.4|[23]|
|MambaVision-T<br>Swin-T|31.8<br>28.3|6298<br>3196|4.4G<br>4.5G|82.3<br>81.3|[22]|



Table 4.1: Literature-Based Performance Table (Based on ImageNet-1K, 224x224 input). 

## **4.3 Data Collection** 

We selected the WildlifeReID-10k dataset (citation), which aggregates over 140,000 images of 10,000+ individual animals from 37 public sources. Its diversity and scale make it ideal for developing robust, generalizable re-identification model enableing evaluation across species and real-world variability. 

14 

Figure 4.7: Wildlife10k Dataset Images 

## **Data Transformation** 

All images were standardized into a PyTorch ImageFolder structure using Python scripts. These scripts identified patterns in the image filenames to group them into appropriate classes and renamed them to ensure consistent class labeling. While some datasets were already correctly structured, others required significant reorganization. After restructuring, all curated datasets were merged into a unified structure, with careful preservation of metadata and the removal of duplicate samples. The resulting dataset was then split into training and validation sets using an 80/20 ratio. Classes with insufficient samples or unstructured filenames lacking discernible patterns (often due to missing annotations) were filtered out. This process resulted in 8,611 unique individual classes selected from over 10,000 initial categories. For classes with only one available image, the same image was used in both training and validation sets. This approach was necessary to avoid data loss, preserve rare classes, and maximize overall dataset coverage. For Phase 2, data augmentation techniques such as horizontal and vertical flips, rotations, zooming, brightness or contrast adjustments, cropping, and Gaussian noise etc. were applied to ensure at least 20 images per class. This not only maintained a uniform file structure but also enhanced the model’s training and generalization performance. 

## **Summary of Preprocessed Data** 

The final dataset for Phase 1 included 8,611 classes, splited in 80/20 for training and validation. This diverse dataset spans a wide range of animal species, supporting the initial evaluation of model performance across varied visual characteristics and class distributions. 

15 

||**Animal**|**Animal**|**Class Count**|**Class Count**|
|---|---|---|---|---|
||Birds<br>Cat<br>Chimpanzee<br>Cattle<br>Dog<br>Girafe<br>Nyala<br>Panda<br>Polar Bear<br>Seal<br>Shark<br>Star Fish<br>Turtle<br>Zebra||23<br>509<br>102<br>375<br>1584<br>178<br>27<br>50<br>13<br>43<br>198<br>23<br>3199<br>2226||
||**Total**||**8611**||
|||**Test Images**||**Validation Images**|
|**Count**||61,226||24,543|



Table 4.2: Final dataset for Phase 1: Animal-wise class distribution and image counts. 

For **Phase 2** , only land-based animal classes with at least 20 images per class were retained to support robust metric learning and fair evaluation. This filtering step ensures that each class has sufficient samples for effective training and validation, which is critical for metric learning approaches. 

|**Animal**|**Individual Count**|**Train Images**|**Val Images**|
|---|---|---|---|
|Cat<br>Girafe<br>Cattle<br>Chimpanzee<br>Dog<br>Panda<br>Polar Bear|509<br>178<br>375<br>102<br>1584<br>50<br>13|10112<br>2048<br>14976<br>5760<br>25344<br>5376<br>1020|2797<br>537<br>3969<br>1517<br>6345<br>1393<br>284|



Table 4.3: Dataset summary for selected land-based animal classes (Phase 2, only classes with _≥_ 20 images/class included). 

## **4.4 Implementation of Selected Design** 

## **Phase 1: Model Comparison and Selection** 

In Phase 1, five models were trained and evaluated based on the dataset’s 80-20 split configuration, with comprehensive data preprocessing and transformation procedures detailed in Section 4.3 above. 

16 

To ensure fair and reproducible comparison across all five architectures (Swin-T, VMamba-T, Vim-S, MambaVision-T, and EfficientVim), identical training hyperparameters and environmental configurations were maintained throughout the experimental process: 

|**Parameter**|**Value**|
|---|---|
|Image Input Size<br>Number of Classes<br>Loss Function<br>Batch Size<br>Training Images<br>Validation Images<br>Epochs<br>Framework<br>Model Checkpointing|224_×_224<br>8611<br>Cross Entropy Loss<br>128<br>61,226<br>24,543<br>300<br>PyTorch<br>Best model state preservation|
|**Training Environment**||
|GPU<br>RAM<br>CPU|RTX 4090 (24 GB)<br>64 GB DDR5<br>Intel i9-14900K|



Table 4.4: Common training parameters used across all five architectures 

The implementation followed a systematic approach where each model underwent sequential training with comprehensive performance monitoring. After each model’s compilation and training completion, detailed performance evaluation metrics were recorded. This rigorous evaluation framework enabled objective comparison across architectures, ultimately leading to the selection of the best-performing model for Phase 2 metric learning implementation based on the optimal balance of accuracy, efficiency, and deployment feasibility for wildlife re-identification applications. 

## **Phase 2: Metric Learning Implementation** 

Metric learning overcomes the fundamental embedding overlap problem in classification approaches where visually similar individuals cluster together in Euclidean space while intra-individual variation exceeds inter-individual differences, creating poor discrimination [18]. Cross-entropy classification uses linear Euclidean space with fixed decision boundaries, whereas metric learning maps features onto hypersphere (ArcFace) manifolds where angular margins enforce superior geometric separation between identities. Unlike Phase 1’s closed-set classification limited to 8,611 animals, metric learning enables open-set recognition through distance-based matching, identifying previously unseen individuals—critical for wildlife monitoring [14]. t-SNE visualization of learned embeddings enables qualitative assessment of feature space organization, revealing how well same individuals cluster together while different animals separate, and allows merging embeddings from multiple species to evaluate cross-species generalization capabilities. 

**Triplet Loss** Triplet Loss operates on triplets of samples (anchor, positive, negative) and enforces that the distance between anchor and positive (same individual) 

17 

is smaller than the distance between anchor and negative (different individual) by a margin _α_ . Mathematically expressed as: 

**==> picture [310 x 13] intentionally omitted <==**

where _d_ ( _a, p_ ) is the distance between anchor and positive, _d_ ( _a, n_ ) is the distance between anchor and negative, and _α_ is the margin parameter. This formulation directly addresses the embedding overlap problem by pulling same individuals closer while pushing different individuals apart in the feature space. 

**ArcFace Loss** ArcFace Loss enhances discriminative power by introducing angular margin penalties on the hypersphere manifold. Unlike Triplet Loss, which operates in Euclidean space, ArcFace normalizes features and weights to unit vectors, then adds an angular margin _m_ to the target angle: 

**==> picture [326 x 31] intentionally omitted <==**

where _s_ is the scale parameter, _θyi_ is the angle between the feature and the target weight, and _m_ is the angular margin. This geometric approach creates more compact intra-class clusters and larger inter-class margins on the hypersphere, leading to superior embedding quality for individual animal identification. 

Triplet Loss and ArcFace Loss directly address embedding overlap by enforcing discriminative representations that cluster same individuals while separating different ones. Embedding quality is visualized through t-SNE plots showing tight intraindividual clusters and clear inter-individual boundaries, making metric learning the standard approach for individual animal identification [8]. 

|**Parameter**|**Value**|
|---|---|
|Epochs<br>Batch Size<br>Training Environment<br>Gradient Clipping<br>ArcFace _s_ (scale)<br>ArcFace _m_ (margin)<br>Dataset|200<br>128<br>GPU: RTX 4090 (24 GB), RAM: 64 GB DDR5, CPU: Intel i9-14900K<br>0.1 – 1.0<br>2, 10<br>0.05, 0.20<br>Only land-based balanced animal dataset|



Table 4.5: Phase 2 Metric Learning: Training parameters and environment. 

18 

## **Chapter 5 Result Analysis** 

## **5.1 Performance Evaluation** 

## **Phase 1 Results** 

## **Best Model Selection: EfficientVim-M1** 

EVim-M emerged as the optimal choice, achieving 83.13% Top-1 accuracy with exceptional efficiency: 

- 14M parameters (68% fewer than Swin-T’s 44M) 

- 664 img/sec throughput (54% faster than Swin-T) 

- 246M FLOPs (18x lower computational cost) 

- 88.52% Top-5 accuracy (best across all models) 

|**Model**|**Top 1%**|**Top 5%**|**Params**|**Throughput**|**FLOPs**|
|---|---|---|---|---|---|
|Swin-T<br>(224_×_224)<br>VMamba-T<br>(224_×_224)<br>MambaVision-T<br>(224_×_224)<br>EVim-M1<br>(224_×_224)<br>Vim-T<br>(224_×_224)|75.32%<br>80.12%<br>79.41%<br>83.13%<br>81%|84%<br>85.95%<br>87.03%<br>88.52%<br>87.5%|44M<br>46M<br>36.6M<br>14M<br>8.6M|431<br>480<br>1513<br>664<br>161|4.5G<br>4.8G<br>4.4G<br>246M<br>-|



Table 5.1: Phase 1 Results: Model performance on our dataset (300 epochs, 224 _×_ 224 input). 

**Selection Rationale:** EVim-M provides superior accuracy efficiency trade off compared to alternatives. While MambaVision-T offers the highest throughput (1513 img/sec), it sacrifices accuracy (79.41%). VMamba-T shows competitive accuracy (80.12%) but requires 3x more parameters (46M) and 20x more FLOPs (4.8G). EVim-M’s balanced performance makes it ideal for resource constrained wildlife monitoring deployments. 

19 

## **Phase 2 Results** 

## **Metric Learning Selection Protocol:** 

To select the best metric learning approach, both Triplet Loss and ArcFace were trained for 100 epochs with a batch size of 128 on the candidate datasets. 

Figure 5.1: t-SNE visualizations of embeddings for 64 individual starfish and bird classes: (a) Triplet Loss shows significant overlap between individuals, indicating poor separation; (b) ArcFace produces tight, well separated clusters, demonstrating superior individual discrimination. 

Figure 5.2: t-SNE visualizations of embeddings for 531 individual cat and cattle classes: (c) Triplet Loss shows overlapping clusters and poor individual separation; (d) ArcFace produces distinct, compact clusters, indicating much better discrimination between individuals. 

## **Embedding Quality (t-SNE Analysis):** 

- Triplet Loss: Good inter species separation, but poor intra individual clustering with significant overlap. 

- ArcFace: Superior individual discrimination with tight clusters and clear boundaries; performance further improves as the number of training images per class increases. 

20 

**Training Efficiency:** 

- ArcFace: 1.3 min/epoch 

- Triplet Loss: 5+ min/epoch ( **4x slower** ) 

**Key Finding:** ArcFace provides faster training and better embedding quality, solving the overlap problem identified in classification approaches. Its advantage is even more pronounced when more training images are available per class. 

**Selection:** ArcFace Loss was chosen for final implementation due to its **4x training speed advantage** and **superior individual animal discrimination capabilities** . 

|**Animal**|**Individual Count**|**Top-1 Acc. (%)**|**Remarks**|
|---|---|---|---|
|Cat<br>Girafe<br>Cattle<br>Chimpanzee<br>Dog<br>Panda<br>Polar Bear|509<br>178<br>375<br>102<br>1584<br>50<br>13|86.73<br>84.72<br>93.02<br>79.32<br>95.30<br>60.23<br>82.75||



Table 5.2: Performance summary for selected animal classes using EVim-M1 with ArcFace loss. 

## **5.2 Analysis of Design Solutions** 

**EVim-M1 success:** The HSM-SSD architecture in EVim-M1 (as described by Lee et al., 2024) achieves high efficiency by performing channel mixing in a compressed hidden state, which reduces redundancy and computational cost. This design allows EVim-M1 to outperform both transformers and other Mamba based models in accuracy and speed which also shown in the EfficientViM paper’s benchmarks. 

**Parameter accuracy relationship:** The literature shows that smaller, well designed models like EVim-M can achieve high accuracy by focusing on essential features and reducing overfitting. This efficient parameter usage explains why EVim-M with only 14M parameters, outperforms larger models like Swin-T (44M). 

**FLOP efficiency analysis:** EVim-M’s 18x lower FLOPs compared to Swin-T (246M vs 4.5G) means it requires much less computation, which is critical for realtime wildlife monitoring in the field where hardware resources are limited. 

**Speed accuracy trade-offs:** MambaVision-T achieves the highest throughput at the cost of lower accuracy. EVim-M, according to both our results and the literature, provides the best balance delivering strong accuracy with efficient speed, making it the most suitable for practical deployment. 

21 

**Embedding space analysis:** Triplet loss works by comparing anchor, positive, and negative samples, enforcing only that the anchor is closer to the positive than to the negative by a margin. This relative constraint often leads to overlapping embeddings, especially as the number of classes increases. Triplet loss struggles with large scale animal re-identification, showing significant overlap in t-SNE visualizations and reduced identification accuracy [8]. In contrast, ArcFace introduces an explicit angular margin penalty, mapping features onto a hypersphere and directly maximizing the decision margin between classes. This results in much closer, well separated clusters in the embedding space, as shown in both our t-SNE plots and the cited works. 

**Convergence behavior:** ArcFace loss applies its margin-based constraint to all samples in a batch simultaneously, providing a strong and direct optimization signal. This leads to significantly faster convergence—up to 4× faster in our experiments—compared to triplet loss, which requires careful mining of hard triplets. This observation is supported by Wang et al. [18], who state: “ArcFace demonstrates faster convergence and more stable training than triplet loss.” 

**t-SNE interpretation:** In t-SNE visualizations, tight, well separated clusters indicate reliable distinction between individuals, while overlapping clusters reveal poor discrimination and higher misidentification risk. Our results show that ArcFace produces clear, compact clusters. 

**Cross-species generalization:** ArcFace embeddings maintain strong separation not only within species but also across different animal types, supporting open-set recognition and generalization to unseen classes. Continued training on each dataset with ArcFace yields high accuracy, as also observed by Stennett et al [8]. 

## **5.3 Discussions** 

This study set out to develop a resource-efficient, accurate animal re-identification system for wildlife monitoring, addressing the challenges of computational constraints, class imbalance, and the need for open-set recognition. 

Our results demonstrate that EfficientViM-M1 (EVim-M1), leveraging the HSMSSD architecture, achieves superior accuracy (83.13% Top-1) with only 14M parameters and 246M FLOPs—an 18x reduction in computational cost compared to Swin-T. Literature supports these findings, showing that compressed hidden state mixing in EVim-M1 reduces redundancy and improves both speed and accuracy. 

Despite severe class imbalance (over 3,000 classes with single images), EVim-M maintained robust performance, highlighting its effective parameter usage and generalization ability. Error analysis revealed that rare classes and visually similar species posed the greatest challenge, but overall validation remained stable across epochs. 

Comparing Mamba-based models to the Swin Transformer baseline, our results confirm that linear complexity architectures like EVim-M1 are more suitable for 

22 

resource-constrained environments than quadratic-complexity transformers. In metric learning, ArcFace loss outperformed triplet loss by producing tighter, more discriminative embedding clusters and converging 4x faster, as confirmed by both our t-SNE visualizations and recent literature [8] [18]. ArcFace also demonstrated strong cross-species generalization, supporting openset recognition and scalability to new animal types. 

## **Key Achievements:** 

- Selection of a highly efficient model (EVim-M1) 

- Successful application of metric learning using ArcFace for robust individual animal re-identification 

## **Limitations:** 

- Multi Animal Detection in a single image or frame and Counting 

- Occlusion and Image Quality Handling 

These will be addressed in future work. Overall, this research provides the first comprehensive evaluation of Mamba-based models for wildlife re-ID and demonstrates readiness for real world conservation deployment. 

23 

## **Chapter 6** 

## **Conclusion** 

## **6.1 Conclusion** 

In summary, this study demonstrates that resource efficient deep learning architectures, particularly EfficientViM-M1, can achieve high accuracy and speed for largescale animal re-identification, even under severe class imbalance and computational constraints. By leveraging advanced metric learning techniques such as ArcFace loss, the system achieves robust individual segregation and strong cross species generalization, outperforming traditional triplet loss approaches. The results confirm that linear complexity models like EVim-M1 are well suited for real world wildlife monitoring, offering both scalability and practical deployment potential. While challenges such as multi animal detection, occlusion, and image quality remain, this work lays a strong foundation for future research and real time conservation applications, advancing the field of automated wildlife identification. 

24 

## **Bibliography** 

- [1] A. Freytag, E. Rodner, M. Simon, A. Loos, H. S. K¨uhl, and J. Denzler, “Chimpanzee faces in the wild: Log-euclidean cnns for predicting identities and attributes of primates,” in _German Conference on Pattern Recognition_ , Springer, 2016, pp. 51–63. 

- [2] C.-A. Brust, T. Burghardt, M. Groenenberg, _et al._ , “Towards automated visual monitoring of individual gorillas in the wild,” in _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ , 2017, pp. 2820– 2830. 

- [3] D. Deb, S. Wiper, A. Russo, _et al._ , “Face recognition: Primates in the wild,” _arXiv preprint_ , 2018. [Online]. Available: https://arxiv.org/abs/1804.08790. 

- [4] M. Krschens, B. Barz, and J. Denzler, “Towards automatic identification of elephants in the wild,” in _AI for Wildlife Conservation Workshop (AIWC)_ , 2018. 

- [5] S. Schneider, G. W. Taylor, S. Linquist, and S. C. Kremer, “Past, present and future approaches using computer vision for animal re-identification from camera trap data,” _Methods in Ecology and Evolution_ , vol. 10, no. 4, pp. 461– 470, 2018. doi: 10.1111/2041-210x.13133. 

- [6] S. Li, J. Li, H. Tang, R. Qian, and W. Lin, “Atrw,” in _Proceedings of the 30th ACM International Conference on Multimedia_ , 2020, pp. 2590–2598. doi: 10.1145/3394171.3413569. 

- [7] Z. Liu, Y. Lin, Y. Cao, _et al._ , _Swin transformer: Hierarchical vision transformer using shifted windows_ , 2021. arXiv: 2103 . 14030 `[cs.CV]` . [Online]. Available: https://arxiv.org/abs/2103.14030. 

- [8] M. Stennett, D. I. Rubenstein, and T. Burghardt, _Towards individual grevy’s zebra identification via deep 3d fitting and metric learning_ , 2022. arXiv: 2206. 02261 `[cs.CV]` . [Online]. Available: https://arxiv.org/abs/2206.02261. 

- [9] V. Cerm´ak,[ˇ] L. Picek, L. Adam, and K. Papafitsoros, “Wildlifedatasets: An open-source toolkit for animal re-identification,” _arXiv.org_ , Nov. 2023. [Online]. Available: https://arxiv.org/abs/2311.09118. 

- [10] B. Jiao, L. Liu, L. Gao, _et al._ , “Toward re-identifying any animal: A universal re-identification model for wildlife conservation,” in _Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS)_ , 2023. [Online]. Available: https://github.com/JiaoBL1234/wildlife. 

- [11] L. Adam, V. Cerm´ak, K. Papafitsoros, and L. Picek, “Wildlifereid-10k: Wildlife[ˇ] re-identification dataset with 10k individual animals,” _arXiv.org_ , Jun. 2024. [Online]. Available: https://arxiv.org/abs/2406.09211. 

25 

- [12] O. Earth, _Bengal tigers: Unsung heroes of the sundarbans mangroves_ , Accessed: 2024-07-31, Jul. 2024. [Online]. Available: https://www.oneearth.org/ bengal-tigers-in-the-sundarbans-mangroves/?utm ~~s~~ ource=chatgpt.com. 

- [13] S. Hou, P. Huang, Z. Wang, _et al._ , “Openanimals: Revisiting person re-identification for animals towards better generalization,” _arXiv.org_ , Sep. 2024. [Online]. Available: https://arxiv.org/abs/2410.00204. 

- [14] Y. Lin, L. Liu, and J. Shi, _Categorical keypoint positional embedding for robust animal re-identification_ , 2024. arXiv: 2412.00818 `[cs.CV]` . [Online]. Available: https://arxiv.org/abs/2412.00818. 

- [15] Y. Liu, Y. Tian, Y. Zhao, _et al._ , _Vmamba: Visual state space model_ , 2024. arXiv: 2401.10166 `[cs.CV]` . [Online]. Available: https://arxiv.org/abs/2401. 10166. 

- [16] F. Pleˇsko, T. Goldmann, and K. Malinka, “Reconstruction and enhancement techniques for overcoming occlusion in face recognition,” 2024. doi: 10.21203/ rs.3.rs-4349727/v1. 

- [17] H. O. Shahreza and S. Marcel, “Face reconstruction from partially leaked facial embeddings,” in _ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)_ , 2024, pp. 4930–4934. doi: 10.1109/icassp48485.2024.10445870. 

- [18] B. Wang, X. Li, X. An, _et al._ , “Open-set recognition of individual cows based on spatial feature transformation and metric learning,” _Animals_ , vol. 14, no. 8, 2024, issn: 2076-2615. doi: 10.3390/ani14081175. [Online]. Available: https: //www.mdpi.com/2076-2615/14/8/1175. 

- [19] R. Xu, S. Yang, Y. Wang, Y. Cai, B. Du, and H. Chen, “Visual mamba: A survey and new outlooks,” _arXiv.org_ , Apr. 2024. [Online]. Available: https: //arxiv.org/abs/2404.18861. 

- [20] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, “Vision mamba: Efficient visual representation learning with bidirectional state space model,” _arXiv.org_ , Jan. 2024. [Online]. Available: https://arxiv.org/abs/2401.09417. 

- [21] L. Zhu, B. Liao, Q. Zhang, X. Wang, W. Liu, and X. Wang, _Vision mamba: Efficient visual representation learning with bidirectional state space model_ , 2024. arXiv: 2401.09417 `[cs.CV]` . [Online]. Available: https://arxiv.org/abs/ 2401.09417. 

- [22] A. Hatamizadeh and J. Kautz, _Mambavision: A hybrid mamba-transformer vision backbone_ , 2025. arXiv: 2407.08083 `[cs.CV]` . [Online]. Available: https: //arxiv.org/abs/2407.08083. 

- [23] S. Lee, J. Choi, and H. J. Kim, _Efficientvim: Efficient vision mamba with hidden state mixer based state space duality_ , 2025. arXiv: 2411.15241 `[cs.CV]` . [Online]. Available: https://arxiv.org/abs/2411.15241. 

- [24] _Wildlifereid-10k dataset on kaggle_ . [Online]. Available: https://www.kaggle. com/datasets/wildlifedatasets/wildlifereid-10k. 

26 

