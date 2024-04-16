

# Title
### Physical-space Multi-body Mesh Detection Achieved by Local Alignment and Global Dense Learning


## envs
```

conda create -n 'demo_3d' python=3.7

conda activate demo_3d

pip install torch==1.8.1+cu102 torchvision==0.9.1+cu102 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install pyglet==1.5.0
pip install scikit-image==0.19.3
pip install lap==0.4.0
pip install seaborn=0.12.2
pip install joblib==1.3.2
pip install pycuda
python -m pip install numpy-quaternion
pip install cython
pip install smplx
pip install seaborn
pip install chumpy
pip install imgaug
pip install loguru
pip install thop
pip install cython_bbox

```


## test

```

python tools/test.py 

```



## env  pip list

```
pip list
Package                       Version              Editable project location
----------------------------- -------------------- -------------------------------------------------------------------------
addict                        2.4.0
alabaster                     0.7.13
appdirs                       1.4.4
Babel                         2.14.0
blessed                       1.20.0
certifi                       2022.12.7
charset-normalizer            3.3.2
chumpy                        0.70
coloredlogs                   15.0.1
coverage                      7.2.7
cycler                        0.11.0
Cython                        3.0.10
cython-bbox                   0.1.5
dill                          0.3.7
docutils                      0.16
exceptiongroup                1.2.0
flatbuffers                   24.3.25
fonttools                     4.38.0
freetype-py                   2.4.0
future                        1.0.0
gpustat                       1.1.1
graphsurgeon                  0.4.5
h5py                          3.8.0
humanfriendly                 10.0
idna                          3.6
imageio                       2.31.2
imagesize                     1.4.1
imgaug                        0.4.0
importlib-metadata            6.7.0
iniconfig                     2.0.0
Jinja2                        3.1.3
joblib                        1.3.2
kiwisolver                    1.4.5
lap                           0.4.0
lmdb                          1.4.1
loguru                        0.7.2
m2r                           0.3.1
Mako                          1.2.4
Markdown                      3.4.4
markdown-it-py                2.2.0
MarkupSafe                    2.1.5
matplotlib                    3.5.3
mdit-py-plugins               0.3.5
mdurl                         0.1.2
mistune                       0.8.4
mmcv-full                     1.4.0                
pmmd_demo_mmcv_v1.4.0
mmdeploy                      0.5.0                
pmmd_demo_mmdeploy_v0.5.0
mmdet                         2.28.2               
pmmd_demo
mpmath                        1.3.0
multiprocess                  0.70.15
myst-parser                   0.18.1
networkx                      2.6.3
ninja                         1.11.1.1
numpy                         1.21.6
numpy-quaternion              2023.0.3
nvidia-ml-py                  12.535.133
onnx                          1.8.0
onnx-graphsurgeon             0.3.12
onnxoptimizer                 0.3.13
onnxruntime                   1.14.1
opencv-python                 4.9.0.80
packaging                     24.0
pandas                        1.3.5
Pillow                        9.5.0
pip                           24.0
platformdirs                  4.0.0
pluggy                        1.2.0
protobuf                      3.20.1
psutil                        5.9.8
pycocotools                   2.0.7
pycuda                        2022.1
pyglet                        1.5.0
Pygments                      2.17.2
PyOpenGL                      3.1.7
pyparsing                     3.1.2
pyrender                      0.1.45
pytest                        7.4.4
python-dateutil               2.9.0.post0
pytools                       2022.1.12
pytorch-sphinx-theme          0.0.24               
pmmd_demo_mmcv_v1.4.0/src/pytorch-sphinx-theme
PyTurboJPEG                   1.7.3
pytz                          2024.1
PyWavelets                    1.3.0
PyYAML                        6.0.1
Quaternion                    3.5.2.post4
realrender                    0.0.5                
realrender
requests                      2.31.0
scikit-image                  0.19.3
scipy                         1.7.3
seaborn                       0.12.2
setuptools                    68.0.0
shapely                       2.0.3
six                           1.16.0
smplx                         0.1.28
snowballstemmer               2.2.0
Sphinx                        4.0.2
sphinx-copybutton             0.5.2
sphinx-markdown-tables        0.0.17
sphinxcontrib-applehelp       1.0.2
sphinxcontrib-devhelp         1.0.2
sphinxcontrib-htmlhelp        2.0.0
sphinxcontrib-jsmath          1.0.1
sphinxcontrib-qthelp          1.0.3
sphinxcontrib-serializinghtml 1.1.5
sympy                         1.10.1
tensorrt                      8.4.0.6
terminaltables                3.1.10
thop                          0.1.1.post2209072238
tifffile                      2021.11.2
tiffile                       2018.10.18
tomli                         2.0.1
torch                         1.8.1+cu102
torchaudio                    0.8.1
torchvision                   0.9.1+cu102
tqdm                          4.66.2
trimesh                       4.2.4
typing_extensions             4.7.1
urllib3                       2.0.7
wcwidth                       0.2.13
wheel                         0.42.0
yapf                          0.40.2
yolox                         0.1.0                
ByteTrack
zipp                          3.15.0
```

## Error Log and solutions:
```
E: The repository 'file:/var/nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8  Release' does not have a Release file.
https://askubuntu.com/questions/1201235/e-the-repository-file-var-nv-tensorrt-repo-cuda10-2-trt7-0-0-11-ga-20191216-r

sudo dpkg -P nv-tensorrt-local-repo-ubuntu2204-8.6.1-cuda-11.8

```

```
replace this code to fix below bugs.
tools/onnx2tensorrt.py
# max_workspace_size=final_params.get('max_workspace_size', 3000), # wrong
max_workspace_size=1 << 30, # correct

Try decreasing the workspace size with IBuilderConfig::setMemoryPoolLimit().
[04/05/2024-23:43:46] [TRT] [E] 1: [resizingAllocator.cpp::deallocate::105] Error Code 1: Cuda Runtime (an illegal memory access was encountered)
[04/05/2024-23:43:46] [TRT] [E] 10: [optimizer.cpp::computeCosts::2033] Error Code 10: Internal Error (Could not find any implementation for node Conv_119 + Add_182.)
Traceback (most recent call last):
  File "tools/onnx2tensorrt.py", line 93, in <module>
    main()
  File "tools/onnx2tensorrt.py", line 84, in main
    device_id=device_id)
  pmmd_demo_mmdeploy_v0.5.0/mmdeploy/backend/tensorrt/utils.py", line 155, in from_onnx
    assert engine is not None, 'Failed to create TensorRT engine'
AssertionError: Failed to create TensorRT engine
```
