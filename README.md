# Defense System for Model Poisoning Attacks in Federated Learning <br>
## Arjun Nair - Capstone Project Usage File <br>

The project was executed and tested on Google Colab Notebook. <br>


The project requires the mxnet package to be installed. <br>
The code automatically downloads the MNIST dataset during execution. <br>

The code has the following variables in the class Args. <br>

dataset = 'mnist'            #dataset <br>
bias = 0.1                   #degree of non-IID <br>
net = 'cnn'                  #model network (Either 'mlr', 'cnn' or 'fcnn') <br>
batch_size = 32              #batch size <br>
lr = 0.0002                  #learning rate <br>
nworkers = 100               #number of clients <br>
nepochs = 100                #number of epochs <br>
gpu = -1                     #gpu 0 if using gpu, -1 using cpu <br>
seed = 41                    #seed <br>
nbyz = 28                    #number of poisoned clients <br>
byz_type = 'partial_trim'    #attack type ('no', 'partial_trim', 'full_trim', 'label_flip', 'backdoor', 'dba') <br>
aggregation = 'median'       #aggregation method ('simple_mean', 'trim', 'krum', 'median') <br>

The only modifications the user has to do is modify these variables with the available options shown and run the code. 
