# Display Virtual Cubes on My Desk!!!
### : 3D Reconstruction and AR via Stereo Vision on Videos  

2023 Spring SNU Computer Vision Project by

Jihwan Kim
kjh26720@snu.ac.kr  
Hojune Kim
hojjunekim@snu.ac.kr  

## How to use
```bash
git clone
conda env create -f "environment.yaml"
conda activate cvproject
cd liegroups
pip install .
cd ..
```
and then,
```bash
python main.py
```
Note that you should click one more point on second plot

## Usage
To verify camera calibration,
```bash
python main.py -c
```
To choose other video,
```bash
python main.py -v="./core/data/short.MOV"
```
To designate output video path,
```bash
python main.py -o="./core/results/output.mp4"
```
To see optical flow processes for each frame,
```bash
python main.py -d
```

Then you can find results in  
./core/results/VIDEO_NAME

## Rule
all files should be located in ./core without main.py  
all images, data files should be located in ./core/data  
all results should be located in ./core/results
