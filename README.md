# AAE4011_GR

Meteor Detection and Counting using YOLOv8 for AAE4011 Group Project.

## Team Members (Group 2)
- Cheung Chun Pang 24036721D
- Chau Hiu Lok 24018496D
- Tang Chung Wang 24020073D


## Introduction

Meteor observation has always been an important part of astronomy, but to monitor the night sky manually for hours is tiring and not very consistent. Human observers may get fatigued, miss faint meteors, and realistically cannot keep watching the sky all day and all night. Because of that, we saw that an automated system using AI could be a more practical solution, for it may monitor continuously and detect meteors in a more consistent manner without the limitations of human observation.

The aim of this project is to build an automated meteor detection and counting system using YOLOv8 object detection. What we are trying to achieve is rather straightforward, to take night-sky video as the input, detect meteors automatically, and then output the counting results together with visualizations that are easy to read. Rather than only training a model on its own, we wanted to make a complete working system that could actually be used, therefore the project also includes a user interface, rather than only consisting of the model development.

According to the original AAE4011 project guideline, the project is expected to apply AI to problems related to unmanned autonomous systems, particularly in areas such as perception, planning, decision-making, or control. Our project mainly falls under the perception category, for the system’s function is to monitor the sky autonomously in a way similar to a camera-based sensing system, detects the object of interest, similar to the AI-powered autofocus functions in the latest digital mirrorless cameras, which for our case our object of interest are meteors, and then outputs structured results for later analysis. In that sense, we believe the project fits the course requirement well, even though the application itself is astronomy-oriented rather than a conventional UAV task.

Our original proposal mentioned using YOLOv5 for detection. However, during implementation we switched to YOLOv8, majorly because it's newer (released January 2023), more accurate, and easier to work with. The Ultralytics YOLOv8 API is cleaner and better documented than YOLOv5, which made our development faster. 

The final system consists of four main parts:
- Dataset download from Roboflow’s public meteor dataset
- Model training using YOLOv8
- Inference and tracking for automated counting
- Graphical user interface built with CustomTkinter

We trained two YOLOv8 variants, namely YOLOv8s and YOLOv8m, tested them on both short meteor compilation clips (few minutes) and long livestream footage (3-8 hours).

During the development process, we have found that meteor detection is practically more difficult than we initially expected, especially when compared with more typical object detection tasks like vehicle detections, for meteors are faint, brief, and relatively rare, while real observatory footage may also contain moonlight, atmospheric noise, lens flare, or even vehicles like aircrafts and satellites, all of which make the detection process less straightforward. Another thing we found is that training on different GPUs, namely the RTX 5070 and RTX 3060, did not only affect the speed, but also the final counting results, which showed us that hardware influence may be more significant than we had first assumed.

This report covers what we have built, how we built it, what worked, what did not work as well as expected, and what we learnt throughout the project. In the following chapters, we will present the dataset, the training process, the system design, the testing results, and the main takeaways from the project as a whole.



---

## Dataset and Preparation

### Dataset Source

We used a public meteor detection dataset from Roboflow called **Meteors v2**, which was exported on 20 June 2024.

Dataset page:  
[Roboflow Meteors v2](https://universe.roboflow.com/dp-detection-of-meteors-and-other-space-objects/meteors-8m2qc)

The dataset is released under the **CC BY 4.0** license, meaning that it may be used for research and educational purposes with proper attribution given. Whereas in our case, the dataset was downloaded automatically through the Roboflow API in Python, therefore we did not need to manually manage a large number of image files one by one.

### Dataset Structure

The dataset contains **747 images** divided into:
- Training Set, which is used to teach the model what meteors look like.
- Validation Set, which is used during training to check if the model is improving
- Test Set, which is kept aside for final evaluation


Each subset includes:
- `images/` for the actual sky images
- `labels/` for annotation text files

The `data.yaml` file specifies the paths and confirms there's only one class in this project:

- `Meteor`

### Image Characteristics

The images come from real observatory footage and were captured using an all-sky camera with a fisheye lens, therefore the data is more realistic but also more difficult to work with, for fisheye lens could cause unnatural deformation of subjects, and reduced image quality at edges. By looking through the sample images, we found that this dataset contains several real-world challenges rather than clean laboratory-style images. 


Main challenges include:
- Fisheye distortion: The wide 180° field of view stretches objects near the edges of the image.
- Low-light conditions:  The night sky is dark, and meteors often appear only as faint streaks.
- Moon interference: Bright moonlight can create lens flare and wash out faint objects, which is visible in many training images.
- Small targets: Meteors usually appear as thin diagonal lines and occupy only a very small portion of the image.
- Sparse meteor events: Some images contain no meteors at all, which reflects the natural rarity of meteor events.

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

From the sample labels we examined, most meteor bounding boxes are relatively small, usually only a small percentage of the full image dimensions, which is consistent with the nature of meteors as tiny and brief targets in the sky viewed from the earth. Some test images also contain empty label files, which means there are no meteors in those frames, and this is expected because meteor events are naturally sparse.

### Preprocessing

According to the Roboflow README, the only preprocessing applied was Auto-orientation, which corrects image rotation based on EXIF metadata. Augmentation was not applied at export time, which means Roboflow didn't artificially expand the dataset with rotations, crops, or colour adjustments. Although our training script adds augmentation later, yet we have intentionally avoided rotation augmentation since meteors are directional tasks, and we would not want to create unrealistic training examples by rotating them.  

---

## Model Training and Setup

### YOLOv8 Variants

For this project, we trained two different YOLOv8 model sizes so that we could compare the trade-off between speed and accuracy:
- **YOLOv8s**
- **YOLOv8m**

In particular, we used YOLOv8s and YOLOv8m, which are both supported in our training setup, and our codebase also shows that the project was built around Ultralytics YOLOv8 rather than the originally proposed YOLOv5 approach.
YOLOv8s, which is the smaller variant, is lighter, faster, and generally more suitable when GPU resources are more limited. YOLOv8m, on the other hand, has more parameters and therefore may give better detection performance, although it also requires more computation during both training and inference.

Both models start from pretrained weights provided by Ultralytics, and from there we fine-tuned them on our meteor dataset. In that sense, we were not training the models entirely from scratch, but rather adapting an already capable object detector so that it could better recognize the visual characteristics of meteor streaks in night-sky footage. This approach is, in our view, more practical for a project like this, because the dataset itself is not extremely large, and transfer learning allows us to make use of previously learned visual features before focusing on the meteor-specific task.

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

The project code explicitly sets degrees=0, fliplr=0.5, and mosaic=1.0, showing that we intentionally kept horizontal flipping and mosaic while avoiding artificial rotation. This was an important decision for us, because meteor detection is not quite the same as ordinary object detection, where random rotation is often acceptable. Meteors are motion streaks with a natural orientation and trajectory, therefore rotating them too much may create image patterns that are unrealistic and not representative of actual observation footage.

For inference:
- Confidence threshold is adjustable through the GUI
- Default confidence is around `0.30` to `0.40`
- IoU threshold is set to `0.5`

We also integrated ByteTrack into the inference pipeline such that meteors can be assigned under unique IDs across consecutive frames, which is important as otherwise the same meteor may be detected multiple times and counted repeatedly. The code for both the script version and the GUI version uses model.track(..., tracker="bytetrack.yaml", conf=..., iou=0.5), which proves that the counting system is based on tracking rather than frame-by-frame detection alone.

### Hardware

Our team trained the models on two different machines, one with a RTX3060, another RTX5070, which meant that the same overall training pipeline was being executed under different hardware conditions. As a result, they have shown a noticeable difference in training speed, which is not surprising given that newer GPUs generally offer stronger performance and better efficiency.

What was more unexpected for us is that the final meteor counts were not always exactly the same when using weights trained on different machines, even when the same test footage was used afterward. At first, this seemed as though there might be an issue in the implementation, but after looking into it further, we came to understand that small numerical differences between hardware platforms may propagate through the detection and tracking process.

This matters especially in our case because the counting logic is not based only on raw detections, but also by ByteTrack assigning IDs over time. When a system like this is slightly sensitive to confidence values, bounding box outputs, or tracking association between frames, even a small difference may eventually lead to a totally different unique count total, even when the visual detections were still looking reasonable overall. So, one of the more important things we learnt here is that hardware reproducibility is not always as straightforward as we first assumed, especially when the pipeline includes sequential components such as tracking rather than a single static prediction step.

For an actual deployed system, such as one intended for long-term observatory use, it would be definitely more appropriate to standardize the entire environment rather than retraining casually across multiple machines. In other words, the exact model weights, software versions, and runtime setup should all be controlled properly, whereas for this project we treated the difference mainly as an observed phenomenon worth documenting, rather than as a failure of the system itself.

---

## System Implementation

### Core Pipeline

The system is built as a complete pipeline rather than only a standalone detector. It includes:
1. Dataset download
2. Model training
3. Inference on video or images
4. Tracking with ByteTrack
5. Output visualization

Our system is built as a complete pipeline rather than only a detection model on its own. In other words, it covers the full process from dataset download, to model training, to inference, tracking, and finally the output of detection results and visualizations. The whole system is built around the Ultralytics YOLOv8 framework, and both the script version and the GUI version follow this general workflow.

The first stage is dataset acquisition. In our case, the dataset is downloaded automatically from Roboflow through the API, where the user provides the API key and the system retrieves the meteor dataset together with the necessary configuration files. This makes the workflow more straightforward, for we do not need to manually sort or manage the dataset files every time the project is set up on another machine.
The second stage is model training. Here, the system loads pretrained YOLOv8 weights, such as YOLOv8s or YOLOv8m, and fine-tunes them on the meteor dataset. During this process, the model gradually learns to distinguish meteor streaks from other image content in the night sky, such as background noise or bright objects, and the training outputs are then saved for later inference use.

The third stage is inference on videos or images. In this stage, the trained model is loaded and utilized to process the selected input, where detections are made based on a chosen confidence threshold. This threshold is important in practice, because setting it too low may increase false positives, while setting it too high may cause faint meteors to be missed.
The fourth stage is tracking, which in our system is handled by ByteTrack. This part is also important because a meteor may appear across several consecutive frames, and without tracking, the system may count the same meteor multiple times. By assigning IDs across frames, the system can count unique meteors in a more reasonable way instead of only counting raw detections frame by frame.

The final stage is the output of results. Our implementation saves annotated detection outputs and also produces a plot showing the number of meteors detected per frame, which gives the user a clearer view of how meteor activity changes across the video. In actual meteor footage, this kind of plot is expected to be sparse, with many frames showing zero detections and occasional peaks when a meteor appears, and that is generally more realistic than having constant detections throughout.
Overall, one thing we find useful about this pipeline is that it is fairly modular. Since dataset download, training, inference, tracking, and plotting are separated in a clear way, where we could simply change individual parts later if needed, such as testing another YOLOv8 variant or adjusting the detection settings, without the need to rebuild the entire system from ground-up.


### GUI

Aside from the script-based workflow, we also developed a graphical user interface using CustomTkinter. The purpose of this GUI is to make the whole system easier to use, especially for users who may not be familiar with running Python code directly from scripts or terminal commands. The project files show that the GUI integrates configuration, training, inference, result plotting, and a shared log window into a single application.

The GUI includes four main tabs:
- Setup
- Training
- Inference
- Results

The Setup tab lets users configure basic paths and settings, including the Roboflow API key, dataset directory path, and model weight file selection. It also displays the current configuration so users can verify their settings before starting training or inference.

The Training tab handles both dataset download and YOLOv8 training. In this tab, the user may choose the model variant, set the epochs, image size, batch size, and confidence threshold, then start the training process directly from the interface. The training information is shown through the shared log output, which helps us monitor whether the training is progressing normally.

The Inference tab is used for running meteor detection on videos and image folders. In our implementation, the user may choose between video mode and image mode, select the relevant source, adjust the confidence threshold using a slider, and then run detection using the selected trained weights. The interface also displays simple statistics such as the total unique meteors and the maximum detections in a frame.

The Results tab is used for visualization. It shows the “meteors detected per frame” plot and permits the plot to be saved as a PNG image, which is helpful when preparing figures for reports or presentations. This part of the system is not overly complicated, but it makes the results easier to interpret than only reading raw console outputs.

One practical advantage of the GUI is that it made testing easier during development. For example, we could adjust the confidence threshold easily and observe how the detection results changed, instead of repeatedly modifying the source code for every small experiment. In that sense, the GUI is genuinely useful for evaluating the trade-off between missing faint meteors and producing false positives.

Overall, the combination of the detection pipeline and the user interface make the system more usable as a complete project. Rather than producing only a trained model, we ended up with something that can download data, train models, run detection, count meteors, and visualize the results in a more organized and practical way.



---

## Testing and Results

### Test Scenarios

Two types of footage were used for evaluation:
- Short meteor compilation clips
- Long livestream-style footage

These two test cases serve different purposes. The compilation clip helps us see whether the system can detect meteors when they appear more frequently, whereas the livestream footage is more useful for checking whether the system can remain quiet during long periods with no meteor activity. In our view, this is important as a good meteor detection system should not keep producing detections all the time, especially when most frames in reality contain nothing. 

### Qualitative Results

These two test cases serve different purposes. The compilation clip helps us see whether the system can detect meteors when they appear more frequently, whereas the livestream footage is more useful for checking whether the system can remain quiet during long periods with no meteor activity. In our view, this is important as a good meteor detection system should not keep producing detections all the time, especially when most frames in reality contain nothing. 

### Observed Hardware Differences 

As mentioned earlier, we noticed that different hardware setups did not only affect training speed, but also caused slight differences in the final counting results. The speed difference itself was expected, but the variation in counts was something we did not initially anticipate.

When the same test footage was processed using weights trained on different machines, the final meteor counts were close but not always exactly the same. This does not necessarily mean one machine is better than the other, but rather that deep learning models and tracking pipelines may be sensitive to small numerical differences. Since our system also uses ByteTrack to assign IDs across frames, even small changes in confidence or detection timing may eventually lead to slightly different unique counts.

### Practical Challenges

Several practical challenges were observed during testing:
- Long videos require significant processing time
- Full frame-by-frame ground truth was not available for long videos
- Confidence threshold strongly affected the balance between false positives and missed detections
- Meteor detection must be interpreted differently from normal object detection, because sparse output is often correct behaviour

From testing, a confidence threshold around `0.3` to `0.4` gave a reasonable balance between sensitivity and false positives.

---

## Limitations and Future Work

One of the main limitations of this project is that our evaluation was mostly qualitative rather than fully quantitative. In practice, we inspected the annotated outputs and the detection plots to judge whether the system was behaving reasonably, but metrics such as precision, recall, or F1 score were not calculated for full-length test videos, mainly because we did not have complete frame-by-frame ground truth annotations for those videos. By that, although we were able to see whether the system appeared to work properly, it was still difficult to compare model settings in a fully objective manner.

Another limitation is that our counting result depends quite heavily on tracking, since the system uses ByteTrack to assign unique IDs across frames rather than simply counting raw detections. The counting workflow is built on YOLOv8 detection together with tracker="bytetrack.yaml" and unique ID accumulation, so even small differences in detections may eventually affect the final count, meaning that although the system is practical, it is not completely immune to small variations in confidence, tracking association, or hardware-related numerical differences.

We also did not implement the Zenithal Hourly Rate estimation, even though it was mentioned in our original project proposal as part of the intended output. Yet, as the project progress our focus has shifted to building and validating the core meteor detection and counting pipeline, as discussed earlier, the counting result of our system still depends heavily on factors such as confidence threshold selection, ByteTrack-based ID assignment, and the hardware used for training and inference, all of which may slightly affect the final meteor count one way or another. Aside from that, given that our evaluation on long videos was mainly qualitative, we considered that implementing ZHR at this stage although would make the system appear more complete, but the resulting astronomical metric would still be built upon counting outputs that were not yet validated rigorously enough. Therefore, we have decided to prioritize the making of detecting tracking, counting, and visualization pipeline work, and leave ZHR estimation as a more suitable extension for another time.

Possible future improvements include:
- Building a properly annotated long-video evaluation set
- Performing more rigorous quantitative evaluation
- Testing reproducibility more systematically across hardware and software environments
- Extending the system to estimate ZHR from timestamps and counts

---

## Conclusion

To conclude, we have developed an end-to-end automated meteor detection and counting system using 2 YOLOv8 variants in this proeject. The system covers the full process from dataset download, to model training, to inference, tracking, and result visualization, and we also developed a GUI so that the project is not limited to command-line use only. The dataset used for training contains 747 images in YOLOv8 format, and this formed the basis of the whole detection workflow.

From our testing, the system was able to process both short meteor compilation clips and longer livestream-style footage in a generally reasonable manner. The detection outputs were sparse instead of continuous, a good sign we expect for meteor observation, for they are brief and rare events. In that sense, the project showed that AI-based automated meteor monitoring is feasible, even though there is still a lot of room for improvement before it could be considered ready for broader real-world use.

Another important part of this project is what we learnt from the implementation process itself. We found that meteor detection is not quite the same as more common object detection tasks, because the targets are faint, short-lived, and infrequent. We also realised that hardware differences may affect not only the speed, but the final behaviour of the pipeline as well, especially when tracking is involved. 

## Project Files
Some result files and the dataset are too large to upload directly to this repository.

- Results and database: [Open SharePoint folder](https://connectpolyu-my.sharepoint.com/:f:/r/personal/24036721d_connect_polyu_hk/Documents/AAE4011/Group?csf=1&web=1&e=gSCWfj)
- Full report (Word): [AAE4011 Group Project Assignment Report.docx](https://github.com/user-attachments/files/26328713/AAE4011.Group.Project.Assignment.Report.docx)

