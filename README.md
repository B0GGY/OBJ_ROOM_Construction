# OBJ_ROOM_Construction

![image](https://github.com/B0GGY/OBJ_ROOM_Construction/blob/master/cover.png)
This is the code for the paper 'From Object to Room: Building Open-Vocabulary Hierarchical Scene Representations'. The mini dataset used for training and validation is included in the 'datasets' dictionary.

## Dependency
You can set up the environment through:
```
conda env create -f environment.yml
conda activate prsclip
```
## Train Head selector
To train the head selector, run:
```
python train.py
```
The model's parameter will be saved in the models_save. We also offered a pretrained model 'model_90.pt' that is used in the paper.
## Room feature combination
To get the result of the room feature combination experiment, run:
```
python room_feature_generator_exp/feature_extractor.py
```
The grandient-based score can be calculated through:
```
python room_feature_generator_exp/head_selector_metrics.py
```
Make sure to run the feature_extractor.py before calculating the score.

## Incremental Spectral Clustering
### Accuracy test
