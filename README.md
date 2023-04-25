# CMCIR
Cross-Modal Causal Relational Reasoning for Event-Level Visual Question Answering     
IEEE Transactions on Pattern Analysis and Machine Intelligence 2023         
For more details, please refer to our paper [Cross-Modal Causal Relational Reasoning for Event-Level Visual Question Answering](https://arxiv.org/abs/2207.12647)     


### Abstract
Existing visual question answering methods often suffer from cross-modal spurious correlations and oversimplified event-level reasoning processes that fail to capture event temporality, causality, and dynamics spanning over the video. In this work, to address the task of event-level visual question answering, we propose a framework for cross-modal causal relational reasoning. In particular, a set of causal intervention operations is introduced to discover the underlying causal structures across visual and linguistic modalities. Our framework, named Cross-Modal Causal RelatIonal Reasoning (CMCIR), involves three modules: i) Causality-aware Visual-Linguistic Reasoning (CVLR) module for collaboratively disentangling the visual and linguistic spurious correlations via front-door and back-door causal interventions; ii) Spatial-Temporal Transformer (STT) module for capturing the fine-grained interactions between visual and linguistic semantics; iii) Visual-Linguistic Feature Fusion (VLFF) module for learning the global semantic-aware visual-linguistic representations adaptively. Extensive experiments on four event-level datasets demonstrate the superiority of our CMCIR in discovering visual-linguistic causal structures and achieving robust event-level visual question answering. 

### Model
![Image](Fig1.png)        
Figure 1: Framework of our proposed CMCIR.        

### Experimental Results
![Image](SUTD.png =100x100)
Figure 2: Results on SUTD-TrafficQA dataset.  
![Image](TGIF.png =100x100)       
Figure 3: Results on TGIF-QA dataset.  
![Image](MSVD.png =100x100)
Figure 4: Results on MSVD-QA dataset.  
![Image](MSRVTT.png =100x100)
Figure 5: Results on MSRVTT-QA dataset.  

### Requirements
- python3.7
- numpy
- pytorch
- [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)    

### Datasets
We conducted our experiment on large-scale event-level urban dataset [SUTD-TrafficQA](https://sutdcv.github.io/SUTD-TrafficQA/#/) and three benchmark real-world datasets [TGIF-QA](https://github.com/YunseokJANG/tgif-qa), [MSVD-QA](https://github.com/xudejing/video-question-answering) and [MSRVTT-QA](https://github.com/xudejing/video-question-answering). The preprocessing steps are the same as the official ones. Please find more details from these datasets.        

### Setups
1. Download [SUTD-TrafficQA](https://sutdcv.github.io/SUTD-TrafficQA/#/), [TGIF-QA](https://github.com/YunseokJANG/tgif-qa), [MSVD-QA](https://github.com/xudejing/video-question-answering) and [MSRVTT-QA](https://github.com/xudejing/video-question-answering) datasets.    
2. Edit absolute paths in preprocess/preprocess_features.py and preprocess/preprocess_questions.py upon where you locate your data.
3. Install dependencies.

## Experiments with SUTD-TrafficQA     
We refer to [SUTD-TrafficQA Official Codes](https://github.com/SUTDCV/SUTD-TrafficQA) for preprocessing.      
### Preprocess Linguistic Features  
1. Download [glove pretrained 300d word vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip) to `/data/glove/` and process it into a pickle file.
```
python txt2pickle.py

```
2. Preprocess train/val/test questions:
```
python 1_preprocess_questions_oie.py --mode train
    
python 1_preprocess_questions_oie.py --mode test
```    
### Preprocess Visual Features    
1. To extract appearance feature with Swin or Resnet101 model:  
 Download Swin [pretrained model](https://github.com/microsoft/Swin-Transformer) (swin_large_patch4_window7_224_22k.pth) and place it to `configs/`.
```
python 1_preprocess_features_appearance.py --model Swin --question_type none

 or
 
python 1_preprocess_features_appearance.py --model resnet101 --question_type none

```

2. To extract motion feature with Swin or ResnetXt101 model:

 Download Swin3D [pretrained model](https://github.com/microsoft/Swin-Transformer) (swin_base_patch244_window877_kinetics600_22k.pth) and place it to `configs/`.
 
 Download ResNeXt-101 [pretrained model](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M) (resnext-101-kinetics.pth) and place it to `data/preprocess/pretrained/`.
```
python 1_preprocess_features_motion.py --model Swin --question_type none

or

python 1_preprocess_features_motion.py --model resnext101 --question_type none

```
### Visual K-means Clustering
1. To extract training appearance feature with Swin or Resnet101 model:  

```
python 1_preprocess_features_appearance_train.py --model Swin --question_type none

 or
 
python 1_preprocess_features_appearance_train.py --model resnet101 --question_type none

```

2. To extract training motion feature with Swin or ResnetXt101 model:

```
python 1_preprocess_features_motion_train.py --model Swin --question_type none

or

python 1_preprocess_features_motion_train.py --model resnext101 --question_type none
```
3. K-means Clustering    
```
python k_means.py
```
Edit absolute paths upon where you locate your data.    

### Training and Testing
```
python train_SUTD.py
```

## Experiments with TGIF-QA    
Depending on the task to chose question_type out of 4 options: action, transition, count, frameqa.
### Preprocess Linguistic Features  
1. Preprocess train/val/test questions:
```
python 1_preprocess_questions_oie_tgif.py --mode train --question_type {question_type}
    
python 1_preprocess_questions_oie_tgif.py --mode test  --question_type {question_type}
```    
### Preprocess Visual Features    
1. To extract appearance feature with Swin or Resnet101 model:  

```
python 1_preprocess_features_appearance_tgif_total.py --model Swin --question_type {question_type}

 or
 
python 1_preprocess_features_appearance_tgif_total.py --model resnet101 --question_type {question_type}

```

2. To extract motion feature with Swin or ResnetXt101 model:

```
python 1_preprocess_features_motion_tgif_total.py --model Swin --question_type {question_type}

or

python 1_preprocess_features_motion_tgif_total.py --model resnext101 --question_type {question_type}

```
### Visual K-means Clustering
1. To extract training appearance feature with Swin or Resnet101 model:  

```
python 1_preprocess_features_appearance_tgif.py --model Swin --question_type {question_type}

 or
 
python 1_preprocess_features_appearance_tgif.py --model resnet101 --question_type {question_type}

```

2. To extract training motion feature with Swin or ResnetXt101 model:

```
python 1_preprocess_features_motion_tgif.py --model Swin --question_type {question_type}

or

python 1_preprocess_features_motion_tgif.py --model resnext101 --question_type {question_type}

```

3. K-means Clustering      

```
python k_means.py
```

Edit absolute paths upon where you locate your data.    

### Training and Testing
```
python train_TGIF_Action.py

python train_TGIF_Transition.py

python train_TGIF_Count.py

python train_TGIF_FrameQA.py
```

## Experiments with MSVD-QA/MSRVTT-QA
### Preprocess linguistic features  
1. Preprocess train/val/test questions:
```
python 1_preprocess_questions_oie_msvd.py --mode train
    
python 1_preprocess_questions_oie_msvd.py --mode test
```    
or    

```
python 1_preprocess_questions_oie_msrvtt.py --mode train
    
python 1_preprocess_questions_oie_msrvtt.py --mode test
```  

### Preprocess visual features    
1. To extract appearance feature with Swin or Resnet101 model:  

```
python 1_preprocess_features_appearance_msvd.py --model Swin --question_type none

python 1_preprocess_features_appearance_msrvtt.py --model Swin --question_type none

 or
 
python 1_preprocess_features_appearance_msvd.py --model resnet101 --question_type none

python 1_preprocess_features_appearance_msrvtt.py --model resnet101 --question_type none

```

2. To extract motion feature with Swin or ResnetXt101 model:

```
python 1_preprocess_features_motion_msvd.py --model Swin --question_type none

python 1_preprocess_features_motion_msrvtt.py --model Swin --question_type none

or

python 1_preprocess_features_motion_msvd.py --model resnext101 --question_type none

python 1_preprocess_features_motion_msrvtt.py --model resnext101 --question_type none

```
### Visual K-means Clustering
1. To extract training appearance feature with Swin or Resnet101 model:  

```
python 1_preprocess_features_appearance_msvd_train.py --model Swin --question_type none

python 1_preprocess_features_appearance_msrvtt_train.py --model Swin --question_type none

 or
 
python 1_preprocess_features_appearance_msvd_train.py --model resnet101 --question_type none

python 1_preprocess_features_appearance_msrvtt_train.py --model resnet101 --question_type none

```

2. To extract training motion feature with Swin or ResnetXt101 model:

```
python 1_preprocess_features_motion_msvd_train.py --model Swin --question_type none

python 1_preprocess_features_motion_msrvtt_train.py --model Swin --question_type none

or

python 1_preprocess_features_motion_msvd_train.py --model resnext101 --question_type none

python 1_preprocess_features_motion_msrvtt_train.py --model resnext101 --question_type none

```
3. K-means Clustering   
```
python k_means.py
```
Edit absolute paths upon where you locate your data.    

### Training and Testing
```
python train_MSVD.py

python train_MSRVTT.py
```

### Citation
If you use this code for your research, please cite our paper.      
```
@article{liu2022cross,
  title={Cross-Modal Causal Relational Reasoning for Event-Level Visual Question Answering},
  author={Liu, Yang and Li, Guanbin and Lin, Liang},
  journal={arXiv preprint arXiv:2207.12647},
  year={2022}
}
``` 
If you have any question about this code, feel free to reach (liuy856@mail.sysu.edu.cn).      
