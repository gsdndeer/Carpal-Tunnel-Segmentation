# Segmentation-of-Carpal-Tunnel-from-magnetic-resonance-image

In recent years, carpal tunnel syndrome(CTS) becomes a common disease due to the heavy load in repetitive wrist work.

The carpal tunnel is the passageway on the palmar side of wrist that connects the forearm to the hand. It is bounded by the transverse carpal ligament and carpal bones, several flexor tendon, median nerve pass through it.

<img src="https://github.com/gsdndeer/Segmentation-of-Carpal-Tunnel-from-magnetic-resonance-image/blob/main/figures/wrist.png" width="550" height="200" >

Magnetic resonance(MR) imaging are widely applied in clinical diagnosis.

Therefore, in this project, I used DeeplabV3+ to segment the wrist tissue (including the flexor tendon, median nerve, and carpal tunnel) from a pair of lateral multimodal (including T1-weighted and T2-weighted) MR images.


## Environment
python 3.6
pytorch


## Get started

1. Clone
```
git clone https://github.com/gsdndeer/Segmentation-of-Carpal-Tunnel-from-magnetic-resonance-image.git
```
2. Download the models from [here](https://drive.google.com/drive/folders/10rljrZTcw8A98jBRzGsSgysMO4S65hOA)

3. Run
```
python predict_gui.py
```

## GUI
<img src="https://github.com/gsdndeer/Segmentation-of-Carpal-Tunnel-from-magnetic-resonance-image/blob/main/figures/gui.png" width="650" height="450">


## Acknowledgement
[Segmentation models](https://github.com/qubvel/segmentation_models.pytorch)
