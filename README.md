# Co-Operative Regressor Networks

![image](https://github.com/user-attachments/assets/ae1c3732-5dcd-4776-b85e-da7e1cb304a5)


#### Train
- Download your training data to the "train_set/" and "val_set" folders respectively.
- Start training
```http
  python CoreNet.py
```
- Training plots will be saved to the "plots/" folder
- Apprentice Regressor weights will be saved to the "weights/" folder
- Master Regressor weights will be saved to the "weightsmaster/" folder
- Start evaluation. You can download Pre-trained Network.
```http
  python test.py
```
- Outputs will be saved to the "test_outputs/" folder 

## Prerequisites
- Pyton 3
- Pytorch
- [FastONN](https://github.com/junaidmalik09/fastonn) 
