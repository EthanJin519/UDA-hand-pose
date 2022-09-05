
# Multi-Branch Adversarial Regression for Domain Adaptative Hand Pose Estimation (MarsDA)

This is the source code for our paper "[Multi-Branch Adversarial Regression for Domain Adaptative Hand Pose Estimation](https://ieeexplore.ieee.org/abstract/document/9732951/metrics#metrics)"

# Citation

If you find our work useful in your research, please consider citing:

	@ARTICLE{jin2022UDAhandpose,
	  title={Multibranch Adversarial Regression for Domain Adaptative Hand Pose Estimation}, 
	  author={Jin, Rui and Zhang, Jing and Yang, Jianyu and Tao, Dacheng},
	  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
	  year={2022}
	}


# Installation

 python_requires='>=3.6'

 Install_requires:

  	 'torch>=1.7.0',
  	 'torchvision>=0.5.0',
  	 'numpy',
  	 'prettytable',
 	  'tqdm',
 	  'scikit-learn',
 	  'webcolors',
 	  'matplotlib',
	  'cv2'.
	  
# Datasets

 [Rendered Handpose Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html) (synthetic dataset)
 
 [Hand-3d-Studio Dataset](https://www.yangangwang.com/papers/ZHAO-H3S-2020-02.html) (real-word dataset)
 
 [Stereo Hand Pose Tracking Benchmark](https://www.dropbox.com/sh/ve1yoar9fwrusz0/AAAfu7Fo4NqUB7Dn9AiN8pCca?dl=0) (real-world dataset) 
 
 
 You need to follow directory structure of the `data` as below.
```
${ROOT}
|-- data
|   |-- RHD
|   |   |-- RHD_published_v2
|   |   |   |-- training
|   |   |   |-- evaluation
|   |-- H3D
|   |   |-- H3D_crop
|   |   |   |-- annotation.json
|   |   |   |-- part1
|   |   |   |-- ...
|   |   |   |-- part5
|   |-- STB
|   |   |-- STB
|   |   |   |-- labels
|   |   |   |-- B1Random
|   |   |   |-- B1Counting
|   |   |   |-- ...
|   |   |   |-- B6Random
|   |   |   |-- B6Counting
```
 
 
 #  Trained models
 
 The trained models can be found [here](https://drive.google.com/drive/folders/1S9NKA2Vvn7XP5D9GB9fwMV9BapV2n-YW?usp=sharing).

 
 
 # Run the code
 
 1. Evaluate on real-world dataset (H3D or STB).
    ```
    python test.py data/H3D -t Hand3DStudio --checkpoint  models/H3D_test.pth,
    python test.py data/STB -t STB --checkpoint  models/STB_test.pth
    ```
   The visualization results will be saved to ``logs``
   
   
 2. Training.
    
    Run the following script:
    ```
    python marsda.py data/H3D -t Hand3DStudio
    python marsda.py data/STB -t STB
    ```
   The pose estimation results will be saved to ``logs``
  
