# Freshness-Analysis-System-for-Fruits-and-Vegetables

This project aims to address this challenge by creating an Edge IoT-based monitoring system for real-time freshness assessment of fruits and vegetables using a Raspberry Pi microcontroller. The Raspberry Pi is equipped with a PiCamera module, which acts as the primary image acquisition component. It captures real-time images of the fruits and vegetables, which are then analyzed by ML algorithms to determine freshness levels. These results are displayed through a user-friendly interface on the Blynk IoT platform, allowing remote monitoring and real-time updates.
The need for such a solution stem from the fact that an estimated 30–40% of post-harvest losses in the fruit and vegetable sector are due to improper handling, poor storage practices, and the inability to accurately assess spoilage in time. 
A cost-effective, intelligent vision-based monitoring system could significantly reduce this wastage, enhance inventory turnover, and boost consumer trust by ensuring only fresh products reach the end user.
Furthermore, this system is highly scalable. Once validated and trained for specific fruits or vegetables, it can be extended to a wider variety of produce with minimal modifications to the software. By leveraging Python libraries such as OpenCV, NumPy, and Scikit-learn, the system performs preprocessing, feature extraction, and classification seamlessly on low-cost embedded hardware. The integration with IoT platforms adds a layer of connectivity that ensures the data is accessible, logged, and usable for future analysis or integration with larger inventory management systems.
This project not only contributes to sustainable food practices but also serves as a foundational step toward more advanced autonomous systems in agriculture, such as robotic sorting systems or automated packaging lines. The fusion of computer vision and IoT in a compact system makes it a practical tool for farmers, distributors, retailers, and quality inspectors, ensuring a positive impact.

Problem Statement: 
Inconsistent and manual inspection of fruits and vegetables often leads to increased food waste, loss of freshness, and consumer dissatisfaction. There is a growing need for automated, scalable, and reliable systems to monitor the freshness and predict the shelf life of perishable items in real time. Our project proposes a smart vision-based solution to bridge this gap using cost-effective technologies

Dataset: https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset
This dataset contains total 12000 images of the following fruits and vegetables items:

Fresh fruits- fresh banana, fresh apple, fresh orange, fresh mango and fresh strawberry.
Rotten fruits- rotten banana, rotten apple, rotten orange, rotten mango and rotten strawberry.
Fresh vegetables- fresh potato, fresh cucumber, fresh carrot, fresh tomato and fresh bell pepper.
Rotten vegetables- rotten potato, rotten cucumber, rotten carrot, rotten tomato and rotten bell pepper.

Methodology
The design and implementation of the Freshness Analysis and Monitoring System of Fruits and Vegetables relies on a well-structured methodology that integrates image processing, machine learning, embedded systems, and IoT-based communication. The system is designed to operate in real-time and on edge devices using low-cost, off-the-shelf components like the Raspberry Pi 4B and PiCamera. This section outlines the step-by-step approach used to build and validate the system.

1. Image Acquisition
The first step involves capturing real-time images of fruits and vegetables using a PiCamera v2 attached to a Raspberry Pi. The resolution is set at 640x480 to balance clarity and processing speed. The setup is placed in a controlled environment with standard lighting to reduce shadows and reflections. Multiple images are taken from different angles and stages of freshness to create a comprehensive dataset. These images form the core of both training and inference phases of the system.

3. Preprocessing
Once the images are captured, they undergo preprocessing using Python libraries such as OpenCV and PIL. This step ensures that the images are normalized and noise-free. Preprocessing includes: Grayscale conversion to reduce computational complexity. Gaussian filtering for noise reduction. Histogram equalization to adjust contrast. Image cropping or resizing for uniformity across the dataset. Preprocessing is crucial as it enhances the key features in the image that the ML model will rely on for decision-making.

3. Feature Extraction
The next step is to extract meaningful visual features from the preprocessed images. The features are the characteristics that help differentiate fresh produce from stale or damaged ones. We use: HOG (Histogram of Oriented Gradients) for shape and edge detection. SIFT (Scale-Invariant Feature Transform) to identify distinctive points that are invariant to scale and rotation. CNN (Convolutional Neural Networks) to automatically learn abstract visual features for freshness classification. These features are either fed directly into machine learning models or used to generate feature vectors for training.

5. Machine Learning Model Training
The system uses supervised machine learning to classify the freshness of fruits. We label the data into categories such as fresh, semi-fresh, and stale based on visual appearance and date of capture. The dataset is split into training and validation sets (80:20). Various algorithms were tested, including: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Logistic Regression, Custom CNN. The CNN model, in particular, demonstrated high accuracy due to its ability to learn complex features. Cross-validation techniques are applied to ensure that the model generalizes well to new data.

6. IoT Integration
To provide remote access and real-time updates, the system is integrated with Blynk IoT. The Raspberry Pi connects to a Wi-Fi network and communicates with the Blynk cloud platform. Freshness predictions are sent from the Pi to the app, where users can view the classification results on their smartphones. The Blynk dashboard is customized with widgets to display current status, timestamp, and freshness score.

7. System Validation
The system is tested using a separate dataset not seen during training. Validation metrics such as accuracy, precision, recall, and F1-score are computed. Real-time trials are conducted under different lighting conditions and with various fruit batches to evaluate performance consistency. The results are compared against benchmark datasets and published studies, confirming the model's reliability and responsiveness.
This methodology ensures a systematic and modular approach to building a smart freshness detection system. It provides the flexibility to scale up, adapt to different produce types, and improve accuracy over time through retraining and dataset expansion.

Elements of the Model: 
The success of the Freshness Analysis and Monitoring System of Fruits and Vegetables relies on the seamless integration of multiple hardware and software elements, each playing a specific role in the real-time processing, analysis, and reporting of freshness data. This section outlines all the core components — both physical and logical — that collectively form the foundation of the system.

1. Raspberry Pi 4B (Model B)
At the heart of the system is the Raspberry Pi 4B, a low-cost, compact, and energy-efficient computer capable of handling advanced computational tasks such as image processing and ML inference. It is equipped with:
•	A 1.5 GHz quad-core ARM Cortex-A72 processor
•	8GB RAM 
•	USB, HDMI, and GPIO interfaces
•	Wi-Fi and Bluetooth connectivity
The Pi acts as both the controller and edge computing node, meaning it handles everything from capturing images to running ML models and transmitting results to the cloud via Wi-Fi.

2. PiCamera Module
The Raspberry Pi Camera Module v2 is used for capturing high-resolution images (640x480) of the fruits and vegetables. It connects to the Pi via a CSI (Camera Serial Interface) port. The camera is mounted in a fixed position and programmed via Python to take periodic or manual snapshots. Key features include:
•	8-megapixel sensor
•	Fixed focus lens
•	Real-time image capture at up to 30fps
Camera settings like exposure, brightness, and ISO are optimized for indoor lighting to ensure image consistency.

3. Python Programming Environment
All the system’s software logic is written in Python, due to its simplicity and extensive library support. The main libraries used include
•	Picamera for controlling the picamera and capturing images
•	OpenCV for image processing (cropping, grayscale conversion, noise filtering)
•	NumPy for array manipulation
•	Matplotlib for visualization during testing
•	scikit-learn and TensorFlow/Keras for building and training ML models
•	PyTorch and Torchvision for generating model files
•	subprocess, os and sys to integrate the inference script with main script, directory declaration, and delay operations respectively.
•	Blynk Library for connecting the Raspberry Pi to the cloud

4. Machine Learning Models
The model used for classification is an ensemble of resnet-18 and efficientnet-b0. It consists of:
•	Convolutional layers for feature extraction
•	Max-pooling layers for down-sampling
•	Fully connected layers for classification
The CNN is trained using a custom dataset of fruit images labelled by freshness levels: fresh and rotten. The model is saved in.joblib format and deployed directly to the Raspberry Pi.
Features of the custom model:
•	Primary tasks: Freshness detection, produce classification, shelf-life prediction
•	Base Models Used: ResNet18, EfficientNet-B0, ShelfLifeModel (custom NN)
•	Machine Learning Models: RandomForestRegressor, GradientBoostingRegressor (for shelf life)
•	Feature Engineering: Multiple feature extractors with intermediate representations
•	Ensemble Approach: Integrated multi-model system
Additionally, traditional ML models like KNN and SVM were also tested during initial stages for benchmarking.

5. IoT Integration with Blynk
Blynk IoT is a low-code IoT platform that simplifies the development and management of connected devices, enabling users to create, deploy, and remotely manage IoT applications from personal projects to commercial products. It is used to build a remote dashboard that allows users to monitor freshness status in real-time. The Raspberry Pi sends data to Blynk using HTTP or MQTT protocols. The mobile app dashboard includes:
•	Label widgets to show classification results (Produce type, Freshness Level, Shelf-Life)
•	Gauge widgets to display confidence scores or freshness percentage
•	Graphs or logs of previous entries
This integration enables remote access and scalability, making the system ideal for retail or warehouse monitoring.

6. Power Supply and Connectivity
A standard 5V/3A USB-C power supply is used to power the Raspberry Pi. For continuous operation in a real-world setting, the system can be powered via portable power banks or integrated into existing power infrastructure. The Pi uses either a local Wi-Fi network or a hotspot to connect to the Internet.
These elements together enable the system to:
•	Operate autonomously
•	Analyze freshness accurately
•	Update users in real-time
•	Scale for different use cases and environments
Each component is selected based on affordability, availability, and performance to ensure the final solution remains accessible for practical deployment.

WORKING
The working of the Freshness Analysis and Monitoring System of Fruits and Vegetables is designed to function as a continuous, real-time, and fully automated process, seamlessly integrating hardware and software elements. The system operates in five main stages: Image Acquisition, Preprocessing, Feature Extraction, Classification, and Output Display, followed by remote transmission using an IoT platform. Each of these stages plays a vital role in ensuring the system delivers accurate, timely, and actionable information about the freshness status of fruits and vegetables.

1. Image Acquisition
The process begins with the Picamera v2, which is attached to the Raspberry Pi. The camera is configured to take high-resolution (640x480 pixels) still images based on user commands. The captured images serve as the primary input for the freshness detection process. The system is typically set up in a controlled lighting environment to ensure minimal noise and better contrast in images with consistent camera angle and distance from the fruit/vegetable across captures.

2. Image Preprocessing
Once the image is captured, it is passed to the preprocessing module written in Python using the OpenCV library. The goal of preprocessing is to standardize the images so that the model only focuses on relevant features. The following steps are performed:	Grayscale Conversion: To reduce complexity by focusing on brightness and contrast instead of color.Noise Reduction: Gaussian blur is applied to smooth out the image. Histogram Equalization: To improve contrast and highlight visual features like spots or bruising. Cropping/Resizing: Ensures uniform image dimensions for model compatibility.
Preprocessing plays a crucial role in making the system adaptable to different lighting environments and ensuring consistent feature extraction.

3. Feature Extraction
After preprocessing, the image undergoes feature extraction. This is where the system identifies key visual cues that are indicative of freshness. The extraction methods used include: HOG (Histogram of Oriented Gradients) to capture edge directions and object shape.	SIFT (Scale-Invariant Feature Transform) to detect unique patterns like spots or blemishes that may indicate spoilage. CNN (Convolutional Neural Networks) automatically extract multi-level features that are not easily hand-engineered.
In the case of CNNs, the input image is passed through a series of convolutional layers that learn to highlight relevant patterns such as discoloration, mold patches, texture inconsistencies, or deformities.

4. Classification
The features extracted are then fed into the model. The model has been trained on a labeled dataset with categories like “Fresh,” and “Rotten”. It performs inference on the new image and returns: Produce Type with confidence level (e.g. 95% confidence),	Freshness with confidence level (e.g. 95% confidence),Shelf-Life Predicted. This classification output is then used to determine the quality status of the produce. The model uses SoftMax activation in the final layer to produce probabilities for each class.

5. Output and IoT Integration
Once the prediction is made, it is displayed both on a local screen (optional) and sent to the Blynk IoT cloud via Wi-Fi. Blynk acts as the remote monitoring interface where users can access and control the system from their smartphones or dashboards. The display includes: A button to capture a photo remotely. A button to stop the device. Text box to show: Predicted Produce (e.g. Bell pepper [91.0% confidence]),	Freshness (e.g. Fresh [94.5% confidence]),Shelf Life (e.g. 8 days).

ANALYSIS
The Analysis phase of the project evaluates the overall performance of the Freshness Analysis and Monitoring System by examining key metrics, comparing results with benchmarks, identifying potential bottlenecks, and validating system reliability under various test conditions. The core aim was to determine the effectiveness of the integrated ML-IoT model in correctly identifying the freshness of fruits using image data captured in real time.

1. Model Performance Metrics
A custom Convolutional Neural Network (CNN) was trained and validated using the pre-processed dataset of fruit images. The classification model was tested for two classes: Fresh, and Rotten. The model performance was evaluated using standard statistical metrics:	Accuracy: 97.33%, Precision: 97.39%, Recall: 97.33%, F1-Score: 97.34%.

3. Real-Time Inference Latency
The time taken for the entire prediction cycle — from image capture to classification result displayed on the Blynk dashboard — was approximately 50-60 seconds. This includes: 0-5 seconds for image capture and pre-processing, 5-60 seconds for inference on the Raspberry Pi, This time can be further reduced using external GPUs or cloud analysis on AWS, GCP and other such platforms when implemented on a larger scale so that the cost incurred by these methods can be justified.

5. Comparison with Research Benchmarks
The performance of our system was compared with results from published literature such as: Yue Yuan and Xianlong Chen (ScienceDirect) [2]: PCA + Deep features (Accuracy: ~91%), Wang et al. (Foods Journal, 2024) [4]: IoT + Spectral Imaging (Accuracy: ~94%), Our model: ResNet18 + EfficientNet-B0+ Custom NN for Shelf Life (Accuracy: ~97%). Our model offered comparable performance using simpler RGB image data and no additional sensors. It demonstrates that basic image-based freshness detection can achieve high reliability when implemented carefully, even without expensive equipment or cloud-based GPUs.

6. Scalability and Edge Performance
The CNN was optimized to run directly on the Raspberry Pi 4B without GPU acceleration for inference only. By resizing images and reducing model complexity (e.g., fewer convolutional layers), real-time inference was possible. While not as fast as cloud solutions, the system demonstrated that: Edge-based inference is viable, Offline operation is possible, Low power and portability are advantages, For larger operations, the same model can be deployed on more powerful hardware or integrated with cloud APIs for high-speed performance.

7. IoT Visualization and Logging
The Blynk dashboard successfully displayed produce type, freshness and shelf-life. The visual analytics component makes this system not just a detection tool, but also a decision-making aid for vendors and managers.

8. Limitations and Future Analysis Enhancements
•	Additional sensory data (e.g., gas concentration, humidity) could improve shelf-life prediction.
•	Dataset size remains a constraint; the system could benefit from 1000+ labelled images per class.
•	Minor improvements in camera calibration and lighting control would enhance consistency.

CONCLUSION
1.The Freshness Analysis and Monitoring System of Fruits and Vegetables marks a significant step forward in the digitization and automation of food quality inspection. In an era where food safety, supply chain efficiency, and sustainability are increasingly becoming global priorities, this system offers a practical and affordable solution that leverages the convergence of machine learning (ML), computer vision, and Internet of Things (IoT) technologies.

2.The primary objective of the project was to design a system capable of evaluating the freshness of fruits and vegetables using real-time image data. Through the use of a Raspberry Pi 4B, Picamera, and a custom-trained CNN model, the system successfully classified produce into various levels of freshness with a reported accuracy exceeding 97%. It also demonstrated stable performance in live environments and was able to deliver predictions within reasonable time, making it suitable for real-time applications in retail, warehouses, and cold storage facilities.
















