# AAE4011_GR

Meteor Detection and Counting using YOLOv8 for AAE4011 Group Project.

## Team Members (Group 2)
- Cheung Chun Pang
- Tang Chung Wang
- Chau Hiu Lok

## Introduction

Meteor observation has long been an important part of astronomy, but continuous manual monitoring of the night sky is tiring and inconsistent. Human observers may become fatigued, miss faint meteors, and cannot realistically observe the sky continuously for extended periods. Because of these limitations, this project explores an AI-based automated solution that can monitor footage continuously and perform meteor detection in a more consistent way.

The aim of this project is to build an automated meteor detection and counting system using YOLOv8 object detection. The system takes night-sky video as input, detects meteors automatically, and outputs counting results together with simple visualizations. In addition to the detection model itself, the project also includes a graphical user interface so that the system can be used more easily in practice.

According to the original AAE4011 project guideline, the work should apply AI to problems related to unmanned autonomous systems, especially in perception, planning, decision-making, or control. This project mainly falls under **perception**, since the system observes visual input, detects meteors as the target of interest, and outputs structured detection results for later analysis. Although the application is astronomy-related rather than a conventional UAV problem, it still aligns well with the perception aspect of the course.

Our original proposal planned to use YOLOv5. However, during implementation we switched to YOLOv8 because it is newer, more accurate, and easier to use. The Ultralytics YOLOv8 API is cleaner and better documented, which helped speed up development.

The final system consists of four main parts:
- Dataset download from Roboflow’s public meteor dataset
- Model training using YOLOv8
- Inference and tracking for automated counting
- Graphical user interface built with CustomTkinter

We trained and tested multiple YOLOv8 variants, including YOLOv8s and YOLOv8m, on both short meteor compilation clips and longer livestream-style footage.

During the development process, we found that meteor detection is in fact more difficult than we initially expected, especially when compared with more typical object detection tasks. Meteors are faint, brief, and relatively rare, while real observatory footage may also contain moonlight, atmospheric noise, and lens flare, all of which make the detection process less straightforward. Another thing we found unexpectedly is that training on different GPUs, namely the RTX 5070 and RTX 3060, did not only affect the speed, but also the final counting results, which showed us that hardware influence may be more significant than we had first assumed.

This report covers what we have built, how we built it, what worked, what did not work as well as expected, and what we learnt throughout the project. In the following chapters, we will present the dataset, the training process, the system design, the testing results, and the main takeaways from the project as a whole.


---

## Dataset and Preparation

### Dataset Source

We used a public meteor detection dataset from Roboflow called **Meteors v2**. It was exported in YOLOv8 format and downloaded automatically using the Roboflow API in Python, which simplified dataset setup and avoided manual file handling.

Dataset page:  
[Roboflow Meteors v2](https://universe.roboflow.com/dp-detection-of-meteors-and-other-space-objects/meteors-8m2qc)

The dataset is released under the **CC BY 4.0** license, which allows research and educational use with attribution.

### Dataset Structure

The dataset contains **747 images** divided into:
- Training set
- Validation set
- Test set

Each subset includes:
- `images/` for the actual sky images
- `labels/` for annotation text files

The `data.yaml` file defines the paths and confirms that the project has only one class:

- `Meteor`

### Image Characteristics

The images come from real observatory footage and were captured using an all-sky camera with a fisheye view, therefore the data is more realistic but also more difficult to work with.
By looking through the sample images, we found that this dataset contains several real-world challenges rather than clean laboratory-style images. 


Main challenges include:
- Fisheye distortion
- Low-light conditions
- Moon interference
- Small targets
- Sparse meteor events

The images are also timestamped, which shows that they were captured from actual night-time observation sessions rather than artificially generated examples, and because of that, we believe this dataset is more suitable for evaluating practical meteor detection performance, although it also makes the training task more challenging.

### Annotation Format

The dataset uses YOLOv8 annotation format. Each label file contains one line per object in the form:

```text
class_id center_x center_y width height
```

Example:

```text
0 0.5429 0.6089 0.0484 0.0822
```

Meaning:
- `0` = Meteor class
- Center at 54.3% of image width and 60.9% of image height
- Width = 4.8% of image width
- Height = 8.2% of image height

From the sample labels we examined, most meteor bounding boxes are relatively small, usually only a small percentage of the full image dimensions, which is consistent with the nature of meteors as small and brief targets in the sky. Some test images also contain empty label files, which means there are no meteors in those frames, and this is expected because meteor events are naturally sparse.

### Preprocessing

According to the Roboflow README, the only preprocessing applied was Auto-orientation, which corrects image rotation based on EXIF metadata. Augmentation was not applied at export time, which means Roboflow didn't artificially expand the dataset with rotations, crops, or colour adjustments. Although our training script adds augmentation later, but we intentionally avoided rotation augmentation since meteors are directional tasks, and we would not want to create unrealistic training examples by rotating them.  

---

## Model Training and Setup

### YOLOv8 Variants

Two YOLOv8 model sizes were trained so that speed and accuracy could be compared:
- **YOLOv8s**
- **YOLOv8m**

YOLOv8s is lighter and faster, making it more suitable for more limited hardware. YOLOv8m is larger and may provide better detection performance, but it also requires more computation.

Both models started from pretrained Ultralytics weights and were then fine-tuned on the meteor dataset. This transfer learning approach is practical for a project with a moderate dataset size, since it allows the model to reuse already learned visual features before adapting to meteor-specific patterns.

### Training Configuration

The training configuration used in the project includes:
- Epochs: `50`
- Image size: `640`
- Batch size: around `8` to `10` depending on GPU memory
- Early stopping patience: `10`
- Pretrained YOLOv8 base weights

For augmentation:
- Horizontal flip enabled
- Mosaic augmentation enabled
- Rotation augmentation disabled

This was a deliberate choice because meteor streaks have natural direction and orientation, so arbitrary rotation could reduce realism.

For inference:
- Confidence threshold is adjustable through the GUI
- Default confidence is around `0.30` to `0.40`
- IoU threshold is set to `0.5`

ByteTrack is integrated into the inference pipeline so that meteors detected in consecutive frames are assigned unique IDs, helping avoid double-counting.

### Hardware

Training was performed on different GPUs, including the **RTX 5070** and **RTX 3060**. As expected, the training speed differed noticeably between systems.

A more surprising result was that the final counting output was not always exactly the same across weights trained on different machines. This suggests that small numerical differences in training and tracking may propagate through the detection pipeline, especially when the final count depends on ByteTrack ID assignment across frames.

---

## System Implementation

### Core Pipeline

The system is built as a complete pipeline rather than only a standalone detector. It includes:
1. Dataset download
2. Model training
3. Inference on video or images
4. Tracking with ByteTrack
5. Output visualization

The dataset is downloaded automatically through the Roboflow API. Pretrained YOLOv8 weights are then fine-tuned on the meteor dataset. During inference, the trained model processes selected videos or image folders using a configurable confidence threshold.

Tracking is performed using ByteTrack so that a meteor appearing across several frames is counted as one event instead of multiple separate detections. The system also generates annotated outputs and a “meteors detected per frame” plot for easier analysis.

### GUI

A graphical user interface was developed using **CustomTkinter** to make the system easier to use without relying entirely on terminal commands.

The GUI includes four main tabs:
- Setup
- Training
- Inference
- Results

The Setup tab handles API keys, dataset paths, and model weight selection. The Training tab controls dataset download and YOLOv8 training. The Inference tab allows users to choose video or image input, adjust the confidence threshold, and run meteor detection. The Results tab shows the generated detection plot and allows it to be saved.

The GUI was useful not only for presentation, but also during development, since it made repeated testing and threshold adjustment much more convenient.

---

## Testing and Results

### Test Scenarios

Two types of footage were used for evaluation:
- Short meteor compilation clips
- Long livestream-style footage

The compilation clips were useful for checking whether the system could detect multiple meteors in a short period. The livestream footage was more realistic and useful for checking whether the system remained quiet during long periods without meteor activity.

### Qualitative Results

For short meteor compilation clips, the system produced sparse detection patterns with occasional spikes, which matched expectations for brief meteor events.

For longer livestream footage, detections were even more sparse. This was considered a positive result, since meteor observation footage should contain many frames with no detections at all. When detections did appear, they were usually associated with visible streak-like events rather than constant false responses to stars or noise.

### Practical Challenges

Several practical challenges were observed during testing:
- Long videos require significant processing time
- Full frame-by-frame ground truth was not available for long videos
- Confidence threshold strongly affected the balance between false positives and missed detections
- Meteor detection must be interpreted differently from normal object detection, because sparse output is often correct behaviour

From testing, a confidence threshold around `0.3` to `0.4` gave a reasonable balance between sensitivity and false positives.

---

## Limitations and Future Work

One limitation of this project is that evaluation was mainly qualitative rather than fully quantitative. We inspected annotated outputs and detection plots, but we did not compute full metrics such as precision, recall, or F1 score for long videos due to the lack of complete frame-by-frame ground truth.

Another limitation is that the counting result depends heavily on tracking. Since the system counts unique IDs produced by ByteTrack, small differences in detections may affect the final meteor count.

We also did not implement **Zenithal Hourly Rate (ZHR)** estimation, even though it was mentioned in the original proposal. Adding this feature would make the system more useful for actual meteor observation analysis, not just detection demonstration.

Possible future improvements include:
- Building a properly annotated long-video evaluation set
- Performing more rigorous quantitative evaluation
- Testing reproducibility more systematically across hardware and software environments
- Extending the system to estimate ZHR from timestamps and counts

---

## Conclusion

This project developed an end-to-end automated meteor detection and counting system using YOLOv8. The system includes dataset download, model training, inference, tracking, result plotting, and a GUI for easier operation.

Testing showed that the system could process both short meteor clips and longer livestream footage in a generally reasonable way. The output was sparse rather than continuous, which is consistent with the real nature of meteor events.

A key lesson from the project is that meteor detection differs from more common object detection tasks because the targets are faint, short-lived, and rare. We also observed that hardware differences may affect not only processing speed but also the final behaviour of the detection and tracking pipeline. Overall, the project demonstrates that AI-based automated meteor monitoring is feasible, while also showing several areas where further improvement is still needed.

## Project Files
Some result files and the dataset are too large to upload directly to this repository.

- Results and database: [Open SharePoint folder](https://connectpolyu-my.sharepoint.com/:f:/r/personal/24036721d_connect_polyu_hk/Documents/AAE4011/Group?csf=1&web=1&e=gSCWfj)
