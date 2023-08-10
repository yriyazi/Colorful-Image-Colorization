import yaml

# Load config file
with open('config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Access hyperparameters
learning_rate       = config['learning_rate']
num_epochs          = config['num_epochs']
seed                = config['seed']
ckpt_save_freq      = config['ckpt_save_freq']


batch_train_size        = config['batch']['batch_train_size']
Validation_Set          = config['batch']['Validation_Set']
batch_validation_size   = config['batch']['batch_validation_size']
batch_test_size         = config['batch']['batch_test_size']


# Access dataset parameters
dataset_path        = config['dataset']['path']
img_height          = config['dataset']['img_height']
img_width           = config['dataset']['img_width']
soft_encode         = config['dataset']['soft_encode']

# Access model architecture parameters
model_name          = config['model']['name']
pretrained          = config['model']['pretrained']
num_classes         = config['model']['num_classes']


# Access optimizer parameters
optimizer_name      = config['optimizer']['name']
weight_decay        = config['optimizer']['weight_decay']
opt_momentum        = config['optimizer']['momentum']
# Access scheduler parameters
scheduler_name  = config['scheduler']['name']
start_factor    = config['scheduler']['start_factor']
end_factor      = config['scheduler']['end_factor']
total_iters     = config['scheduler']['total_iters']

# print("configuration hass been loaded!!! \n successfully")
# print(learning_rate)