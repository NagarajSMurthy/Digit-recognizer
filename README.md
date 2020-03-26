# Digit-recognizer
Digit recognition is a simple task of recognising digits from 0 to 9. A CNN is trained on the MNIST dataset which gave a good accuracy of around 98% on the test set. Choosing the proper model and employing suitable regularisation technique, the model also performed well during inference with an accuracy of close to 98%. The whole model is then converted to ONNX format and run on CPU with ONNX Runtime.


### Usage
1. Train the model using the 'MNIST classification using ConvNets' notebook. Save your models by specifying the path. 
2. Select the model that you want to convert to ONNX format. Specify the path to your saved model in the MNIST_ONNX notebook. Save your model in the onnx format (say, model.onnx).   
3. Load the model.onnx file and run an inference session on your CPU using ONNX Runtime. Make sure to specify the path in mnist_onnx.py 

You can directly download the model in onnx format here https://drive.google.com/file/d/16y9_jh7H7La0M1vuQ3XcpdpYu-ibJbJB/view?usp=sharing. 


![](Mnist_onnx.mp4)
