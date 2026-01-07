# Pixel-play--26-submission-Soham
it consist of 2 models solving the kaggle pixel play'26 contest ,3 csv files and report
VLG Recruitment Challenge ‘26 Report
SOHAM JAIN (25125035, WHATSAPP NO. 7276427190)

Introduction
This project aims to develop a model to detect anomalous events in surveillance footage. The task poses unique challenges, particularly the requirement to accurately identify irregularities using the provided video dataset. The task involves training a model to analyse sequential video frames and assign probability scores to potential anomalies. The focus is on generalization and the ability to distinguish between standard background activity and significant deviations in a real-world environment.

Model Development

1. Data Preprocessing
MODEL 1 – Optical Flow + HOG + Isolation Forest
a) Image preprocessing
All the images were transformed from 640×320 RGB to 256×256 grayscale for higher-speed .
Most importantly, all the frames whose rotation was wrong were rotated vertically upside. The wrong frames were detected by comparing energy in the top half and bottom half. If the top half had lesser energy than the bottom half, then the frame was rotated vertically up.
b) Data restructuring and handling
All the frames of one video were kept differently so that they do not mix with other videos. Frames were stored in structured dictionaries with ordered frames (grayscale 256×256) and with their respective frame IDs.

MODEL 2 – ResNet + Flow + Isolation Forest + K-Means
1. Data Preprocessing
a) Image preprocessing
All the images were transformed from 640×320 RGB to 256×256 grayscale for higher-speed processing.
Most importantly, all the frames whose rotation was wrong were rotated vertically upside. The wrong frames were detected by comparing energy in the top half and bottom half. If the top half had lesser energy than the bottom half, then the frame was rotated vertically up.
Two parallel formats of each frame were stored:
• Grayscale frames used for optical flow and motion modeling
• RGB tensors (ImageNet normalized) used for ResNet-18 deep feature extraction
b) Data restructuring and handling
Frames were stored video-wise inside structured dictionaries to prevent cross-video mixing.
For each video, the following were maintained:
• ordered grayscale frames
• ordered RGB tensors
• aligned frame IDs

2. Model Architecture

MODEL 1 – HOG + Flow + Isolation Forest
a) Appearance features (128)
HOG (Histogram of Oriented Gradients) was used to capture edges, object shapes, and patterns. I divided each frame into small cells (8×8 pixels) and applied HOG. Thus, we get 32×32 total cells and HOG uses a 2×2 blocking strategy. So, we get total blocks = 31×31, and then for each block we had 2×2×9 cells (because 9 orientations were used). So, we get total features = 34,596 per frame.
34,596 is a huge number to compute and train, so I applied PCA and used only the 128 most useful features.
b) Motion features (21)
Dense optical flow (Farneback) was used to capture pixel-level motion, which gave vx, vy and then I converted it to polar coordinates (speed and angle). 8 distinct magnitude features were extracted, 8 direction features were extracted, 1 entropy-based feature, 1 coherence feature, and 3 acceleration features were extracted.
In total, we get 21 motion-related features.
c) Temporal enhancement (63 = 21×3 optical features expansion)
Observation: Some frames whose size >130 kb were having very high pixel-level noise and vignette. To avoid this, I used temporal processing (smoothing and temporal difference).
Process: used a short window of 3 and averaged the temporal features.
Multi-scale temporal deltas: delta(t-1) and delta(t-5) were used for detecting sudden and gradual irregularities. So final enhanced motion vector = 63D per frame.
d) Feature fusion
128 appearance and 63 motion features were combined to 191D total features per frame, then because they were of different scales and magnitudes, I used StandardScaler (zero mean, var = 1).
But .fit only on train and .transform to both train and test to protect data leakage.
e) Anomaly model
Isolation Forest was used for anomaly detection. The idea was that all anomalies will be isolated very quickly at lesser depths.

MODEL 2 – ResNet + Flow + Isolation Forest + K-Means
Model-2 follows a dual-branch architecture:
a) Flow features (21D)
Dense optical flow (Farneback) was used between consecutive frames to obtain pixel-level motion vectors. vx and vy obtained were converted to polar coordinates (magnitude and angle).
From these, 21 motion features per frame were extracted:
• 8 magnitude statistics
• 8 direction histogram
• 1 entropy feature
• 1 coherence feature
• 3 acceleration features
These features describe overall motion.
They were standardized using StandardScaler and trained using Isolation Forest to learn normal motion behavior.
Unlike Model-1, Isolation Forest here is not the final anomaly detector, but acts as one of the two gates used for anomaly scoring.

b) Deep semantic branch (512D)
A pretrained ResNet-18 model was used as a semantic feature extractor. The default classification layer was removed, and each frame got 512-dimensional features.
These features  capture:
• object presence
• posture
• spatial information
Temporal max-pooling was applied over a local window of size 8.

c) K-Means clustering
Instead of learning a single normal center, K-Means clustering (k = 5) was applied to the training semantic features.
This allowed the model to represent different normal behaviors such as:
• walking left
• walking right
• moving towards camera
• moving away
• standing/background
Anomaly score was defined as the distance to the nearest normal cluster center, ensuring that a frame is considered normal if it matches any valid normal behavior mode.

d) Final anomaly scoring
Final anomaly score was computed as:
Final score = Semantic anomaly × Sigmoid(Motion anomaly)
A frame is anomalous only if it looks abnormal (high semantic anomaly) and moves abnormally (high motion anomaly). It helps to reduce false positives which was seen in graphs and also AUC.
Global min-max normalization was applied after fusion to produce final anomaly probabilities.

3. Training

MODEL 1 – Optical Flow + HOG + Isolation Forest
Loss Function / Training Objective
I used Isolation Forest, which does not have an explicit loss function. Random forests are based on the objective to use various random isolation trees that continuously  partition the feature matrix until a point becomes isolated in a leaf node. The core idea is anomalies are easier to isolate than normal points.
Anomaly scores were obtained using the decision_function method of Isolation Forest, which measures the deviation of a sample’s path length from the expected normal behavior, thus giving a regularity score. This regularity score was multiplied by -1 and finally normalized per video using min-max method and then it was used as anomaly score.
Optimizer
No explicit optimizer used. Isolation Forest is a tree-based ensemble method. Training consists of random subsampling and random feature selection.
Epochs and Batch Size
Not applicable. All frames from the training videos are treated as samples of normal behavior, and each isolation tree is built using random subsamples of this dataset.
Hardware
Kaggle notebook was used and 2×T4 for faster speed.
Training Details
PCA was fitted only on training HOG and not on test HOG, and the corresponding most important 128 features derived from training HOG were used to transform both test and training original HOG features.
Isolation Forest was trained only on training data using 191 total extracted features (128 HOG and 63 flow-based) with following hyperparameters: 400 estimators, contamination = 0.1, max_features = 1.0.

MODEL 2 – ResNet + Flow + Isolation Forest + K-Means
Loss Function / Training Objective
Like Model-1, Model-2 also does not use a conventional loss function.
Isolation Forest isolates abnormal motion. K-Means minimizes intra-cluster variance in the ResNet embedding space, forcing semantic normal behaviours to form compact clusters.
Optimizer
No explicit optimizer was used.
Epochs and Batch Size
Not applicable.
• Isolation Forest and K-Means are fit once on the full training feature set.
• ResNet inference used batch processing (batch size 64) only for memory safety.
Hardware
Kaggle notebook was used and 2×T4 for faster speed.
Training Details
All fitting stages in Model-2 were performed strictly on the training set.
A pretrained ResNet-18 with temporal max-pooling was applied for short-term context. K-Means clustering (k = 5) was fitted only on training data to learn multiple normal semantic modes. These learned centers were used to compute anomaly scores for both training and test frames.
Dense optical flow features were standardized using a StandardScaler fitted only on training motion features. Isolation Forest was trained only on the training motion features with 300 estimators, contamination = 0.10, max_features = 1.0. The trained model was then used to score both training and test motion data.
Final anomaly scores were computed using gated fusion between semantic distance scores and sigmoid-transformed motion anomaly scores normalization was applied only after fusion.

Results
Model 1
The natural CSV generated using the code for Model-1 gave AUC  0.50, and when some boosting was applied it gave 0.55. I have attached the code for boosting as the last cell of notebook
Model 2
The natural CSV generated using the code for Model-2 gave AUC 0.57, and when boosting was applied it gave 0.61. I have attached the code for boosting as the last cell of notebook


Conclusion
On comparing Model-1 and Model-2, we can see from the graph that Model-2 identifies positives with more confidence and thus reduces false negatives(anamoly getting classified as normal).
 
(graph by gemini by giving csv files to it)
From the second graph, we can see that Model-2 is more aggressive as it has more density towards higher scores.
 
As these two models were quite different, taking the weighted average gave the highest AUC, approximately 0.65.

Challenges
At the start of the competition, I didn’t know neural networks much, so I restricted myself to only machine learning algorithms I know well . I started with one-class SVM, but it failed badly because I was not using flow features and I had not rotated wrongly flipped frames.
Then I tried Isolation Forest with flow and HOG but resized frames to 64×64, which led to low AUC due to loss of motion information. Finally, I found 256×256 to be the most suitable.
Then I learned about ResNet and tried to use it with kmeans , which also failed. Finally, combining ResNet, flow, and Isolation Forest improved performance. Weighted average of both models achieved the best result.

Learning Outcomes
1)	Learned how to build an end to end complete ml project
2)	Learned how to be better at data preprocessing
3)	Learned about ResNet
4)	Tried to build many graphs and learned about data analysis
5)	Learned what to do when things don’t work
