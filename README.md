# CMCIR
Cross-Modal Causal Relational Reasoning for Event-Level Visual Question Answering     
Preprint 2022           
For more details, please refer to our paper [Cross-Modal Causal Relational Reasoning for Event-Level Visual Question Answering](https://arxiv.org/abs/2207.12647)     

<a href="https://orcid.org/0000-0002-9423-9252" target="orcid.widget" rel="noopener noreferrer" style="vertical-align:top;"><img src="https://orcid.org/sites/default/files/images/orcid_16x16.png" style="width:1em;margin-right:.5em;" alt="ORCID iD icon">orcid.org/0000-0002-9423-9252</a>

Homepage: [https://yangliu9208.github.io/home/](https://yangliu9208.github.io/home/)

### Abstract
Existing visual question answering methods tend to capture the spurious correlations from visual and linguistic modalities, and fail to discover the true casual mechanism that facilitates reasoning truthfully based on the dominant visual evidence and the correct question intention. Additionally, the existing methods usually ignore the complex event-level understanding in multi-modal settings that requires a strong cognitive capability of causal inference to jointly model cross-modal event temporality, causality, and dynamics. In this work, we focus on event-level visual question answering from a new perspective, i.e., cross-modal causal relational reasoning, by introducing causal intervention methods to mitigate the spurious correlations and discover the true causal structures for the integration of visual and linguistic modalities. Specifically, we propose a novel event-level visual question answering framework named Cross-Modal Causal RelatIonal Reasoning (CMCIR), which consists of three essential components named causality-aware visual-linguistic reasoning module, spatial-temporal transformer, and visual-linguistic feature fusion module, to achieve robust casuality-aware visual-linguistic question answering. To uncover the causal structures for visual and linguistic modalities, the novel Causality-aware Visual-Linguistic Reasoning (CVLR) module is proposed to collaboratively disentangle the visual and linguistic spurious correlations via elaborately designed front-door and back-door causal intervention modules. To discover the fine-grained interactions between linguistic semantics and spatial-temporal representations, we build a novel Spatial-Temporal Transformer (STT) that builds the multi-modal co-occurrence interactions between visual and linguistic content. To adaptively fuse the causality-ware visual and linguistic features, we introduce a Visual-Linguistic Feature Fusion (VLFF) module that leverages the hierarchical linguistic semantic relations as the guidance to learn the global semantic-aware visual-linguistic representations adaptively. Extensive experiments on large-scale event-level urban dataset SUTD-TrafficQA and three benchmark real-world datasets TGIF-QA, MSVD-QA, and MSRVTT-QA demonstrate the effectiveness of our CMCIR for discovering visual-linguistic causal structures and achieving robust event-level visual question answering.

### Model
![Image](Fig1.png)        
Figure 1: Framework of our proposed CMCIR.        

### Requirements
- python3.7
- numpy
- pytorch
- [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)    

### Datasets
We conducted our experiment on large-scale event-level urban dataset [SUTD-TrafficQA](https://sutdcv.github.io/SUTD-TrafficQA/#/) and three benchmark real-world datasets [TGIF-QA](https://github.com/YunseokJANG/tgif-qa), [MSVD-QA](https://github.com/xudejing/video-question-answering) and [MSRVTT-QA](https://github.com/xudejing/video-question-answering). The preprocessing steps are the same as the official ones. Please find more details from these datasets.        

### Setups
Code will be released when it is fully prepared, please stay focusing on!   
1. Dowanload [SUTD-TrafficQA](https://sutdcv.github.io/SUTD-TrafficQA/#/), [TGIF-QA](https://github.com/YunseokJANG/tgif-qa), [MSVD-QA](https://github.com/xudejing/video-question-answering) and [MSRVTT-QA](https://github.com/xudejing/video-question-answering) datasets.    
2. Edit absolute paths in preprocess/preprocess_features.py and preprocess/preprocess_questions.py upon where you locate your data.
3. Install dependencies.

## Experiments with SUTD-TrafficQA     
### Preprocess Linguistic Features  
1. Download [glove pretrained 300d word vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip) to `data/glove/` and process it into a pickle file.

2. Preprocess train/val/test questions:
```
python 1_preprocess_questions_oie.py --mode train
    
python 1_preprocess_questions_oie.py --mode test
```    
### Preprocess Visual Features    
1. To extract appearance feature with Swin or Resnet101 model:  

```
python 1_preprocess_features_appearance.py --model Swin --question_type none

 or
 
python 1_preprocess_features_appearance.py --model resnet101 --question_type none

```

2. To extract motion feature with Swin or ResnetXt101 model:

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
If you have any question about this code, feel free to reach me (liuy856@mail.sysu.edu.cn)      
