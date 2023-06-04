Arjun Nair - Capstone Project Usage File

The project was executed and tested on Google Colab Notebook.


The project requires the mxnet package to be installed.
The code automatically downloads the MNIST dataset during execution.

The code has the following variables in the class Args. 

dataset = 'mnist'            #dataset
bias = 0.1                   #degree of non-IID
net = 'cnn'                  #model network (Either 'mlr', 'cnn' or 'fcnn')
batch_size = 32              #batch size
lr = 0.0002                  #learning rate
nworkers = 100               #number of clients
nepochs = 100                #number of epochs
gpu = -1                     #gpu 0 if using gpu, -1 using cpu
seed = 41                    #seed
nbyz = 28                    #number of poisoned clients
byz_type = 'partial_trim'    #attack type ('no', 'partial_trim', 'full_trim', 'label_flip', 'backdoor', 'dba')
aggregation = 'median'       #aggregation method ('simple_mean', 'trim', 'krum', 'median')

The only modifications the user has to do is modify these variables with the available options shown and run the code.