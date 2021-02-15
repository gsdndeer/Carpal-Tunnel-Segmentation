# Segmentation-of-Carpal-Tunnel-from-magnetic-resonance-image

In recent years, carpal tunnel syndrome(CTS) becomes a common disease due to the heavy load in repetitive wrist work.

The carpal tunnel is the passageway on the palmar side of wrist that connects the forearm to the hand. It is bounded by the transverse carpal ligament and carpal bones, several flexor tendon, median nerve pass through it.

<img src="https://github.com/gsdndeer/Segmentation-of-Carpal-Tunnel-from-magnetic-resonance-image/blob/main/figures/wrist.png" width="550" height="200" >

Magnetic resonance(MR) imaging are widely applied in clinical diagnosis.



## Method
In this project, I used DeeplabV3+ to segment the flexor tendon, median nerve, and carpal tunnel separately from a pair of lateral multimodal (including T1-weighted and T2-weighted) MR images. 



## Environment
1. Python 3.6.8
2. Pytorch 1.7.0



## Get started

1. Clone the repository
```
git clone https://github.com/gsdndeer/Segmentation-of-Carpal-Tunnel-from-magnetic-resonance-image.git
```
2. Download the data from [here](https://drive.google.com/drive/folders/1clUZVY3Vc4jX179rUZQdAMyN6nbN1eB4?usp=sharing)

3. Download the models from [here](https://drive.google.com/drive/folders/10rljrZTcw8A98jBRzGsSgysMO4S65hOA)

4. Run
```
python predict_gui.py
```


## GUI
<img src="https://github.com/gsdndeer/Segmentation-of-Carpal-Tunnel-from-magnetic-resonance-image/blob/main/figures/gui.png" width="650" height="450">



## Acknowledgement
1. [Segmentation models](https://github.com/qubvel/segmentation_models.pytorch)
2. The data is provided by [NCKU CSIE VSLAB](https://sites.google.com/view/ncku-csie-vslab)
