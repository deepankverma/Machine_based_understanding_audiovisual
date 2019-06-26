# Extraction of Environmental Attributes from manually collected street-level Audio and Visual data.
## AUDIO CLASSIFICATION ## 

 - Requirements:
	 - Python 3.6
	 - Tensorflow 1.7
	 - Keras 2.2.0

The complete AudioSet data consists of 632 classes. To classify street-level sounds, A subset of AudioSet data is prepared which
consists of 16 classes. The download options are provided in the [AudioSet](https://research.google.com/audioset/download.html) website.
In this project, unbalanced_train_segments and balanced_eval are utilized for model training and evaluation respectively.

- The audio_dataset_analysis folder consists of 4 Jupyter notebooks:

	 - Data_preparation : Preparation of subset of AudioSet data.
   - Training : LSTM Model training with Keras and Tensorflow.
   - Evaluation : Evaluation of Model performance with overall accuracy and Kappa metrics.
   - Inference : Model inferences on manually collected audio data.


## VISUAL CLASSIFICATION ## 

Visual Classification includes Object Detection (OD) and Semantic Segmentation (SS) task. 

 - Requirements:
	 - Python 3.6 (OD), Python 3.5 (SS)
	 - Tensorflow 1.7 (OD), Tensorflow 1.2.1 (SS) 
	 - Keras 2.2.0
   - MongoDB 3.2.19
   
Both the tasks can be executed in the terminal.

### Object Detection ### 
`python object_detect.py -g "your_folder_where_images_are_stored/*.jpg" -c "name_of_db_collection" --id 1`

### Semantic Segmentation ###  
`python pspnet_2.py -m pspnet50_ade20k -g "your_folder_where_images_are_stored/*.jpg" -o "output_folder_to_store_segmented_images"  -c "name_of_db_collection" --id 1`

where `--id 1` refers to the use of CPU, replace it with `0` to use GPU (Requires CUDA to be installed and configured with Tensorflow) 

### RESEARCH
This code is a part of published paper in journal of [Landscape and Urban Planning](https://www.sciencedirect.com/science/article/pii/S0169204618313835).


### Reference Repositories ###

 - https://github.com/ideo/LaughDetection
 - https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow
 - https://github.com/tensorflow/models/tree/master/research/object_detection
