# Synthesis Pipeline
This is an implemented pipeline for generating synthetic images, depths and surface normals from [ShapeNet](https://www.shapenet.org/). You can modify `cfg_global.yaml` to set up the environment.

## Installation
One-button initialization (downloading shapenet dataset, sun2012pascal dataset, view distribution and blender-2.71). Comment the code for downloading the dataset if needed.
```
sh init.sh
```
You must download blender-2.71 for this tool (as in the script). (Yes! the specific version of blender)

## Usage
```
python render_random.py -p ${PARAM_FILENAME} -o ${OUTPUT_FOLDER} -n ${CLASS_NAME}
```
A simple example is given:
```
python render_random.py -p testcase1 -o testcase1 -n sofa
```

The parameter file is stored in `paramGen` folder. You can follow the style of any one file to pass in the parameters for rendering your own synthetic dataset.

## Re-rendering the dataset of 3D-R2N2
We provide a script to re-render the images provided by the authors of [Choy et al., 3D-R2N2: A Unified Approach for Single and Multi-view 3D Object Reconstruction, ECCV 2016](http://cvgl.stanford.edu/3d-r2n2/) with depth images and surface normals. You should first download the original dataset via:
```
wget http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
tar -xzf ShapeNetRendering.tgz
```
Then, you could setup the path and run our scripts:
```
python render_3dr2n2_25d.py
```


