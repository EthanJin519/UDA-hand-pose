
Multi-Branch Adversarial Regression for Domain Adaptative Hand Pose Estimation

Installation：
python_requires='>=3.6',
install_requires=[
   'torch>=1.7.0',
   'torchvision>=0.5.0',
   'numpy',
   'prettytable',
   'tqdm',
   'scikit-learn',
   'webcolors',
   'matplotlib'
],

Dataset：
  Rendered Handpose Dataset(synthetic dataset)，
  Hand-3d-Studio Dataset(real-word dataset)，
  Stereo Hand Pose Tracking Benchmark(real-world dataset).



Running the code
1. Evaluate on real-world dataset H3D or STB
   python test.py data/H3D -t Hand3DStudio --checkpoint  models/H3D_test.pth,
   python test.py data/STB -t STB --checkpoint  models/STB_test.pth
   
2. Training
   python marsda.py data/H3D -t Hand3DStudio
   python marsda.py data/STB -t STB
   
  
