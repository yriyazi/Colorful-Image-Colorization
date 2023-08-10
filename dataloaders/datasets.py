import  os
import  glob
import  torch
import  utils
import  numpy                   as      np
import  sklearn.neighbors       as      nn
import  torchvision.transforms  as      transforms
from    skimage                 import  color
from    torch.utils.data        import  Dataset
from    PIL                     import  Image


def crawler(DIR_TRAIN=utils.dataset_path):
    """
    Crawls through a specified directory and returns a list of image addresses.

    Args:
        DIR_TRAIN (str): The directory path to crawl through.
        Default is 'datasets/landscapes'.

    Returns:
        list: A list of image addresses found in the directory.
    """
    Fileames = os.listdir(DIR_TRAIN)
    Images_adress  = []
    for _FIlename in Fileames:
        Images_adress  += glob.glob('datasets/landscapes/'  + _FIlename )
    print("\nTotal train images: ", len(Images_adress))
    
    return Images_adress

def crawler_hand_picked(path:str = 'Data/Directories.txt',):
    # Open the file in read mode
    with open(path, 'r') as file:
        # Read all lines of the file into a list
        lines = file.readlines()
    # Remove newline characters from each line and create a list
    return [line.strip() for line in lines]

def preprocess_img(Input_image:np.ndarray):
    Resized_RGB_LAb                     = color.rgb2lab(Input_image)
    Resized_RGB_LAb_Lumination          = Resized_RGB_LAb[:,:,0 ]
    Resized_RGB_LAb_ab                  = Resized_RGB_LAb[:,:,1:]
    return Resized_RGB_LAb_Lumination,Resized_RGB_LAb_ab

def get_soft_encoding(image_ab, nn_finder, nb_q):
    h, w    = image_ab.shape[:2]
    a       = np.ravel(image_ab[:, :, 0])
    b       = np.ravel(image_ab[:, :, 1])
    ab      = np.vstack((a, b)).T
    # Get the distance to and the idx of the nearest neighbors
    dist_neighb, idx_neigh = nn_finder.kneighbors(ab)
    
    # Smooth the weights with a gaussian kernel
    sigma_neighbor = 5
    wts = np.exp(-dist_neighb ** 2 / (2 * sigma_neighbor ** 2))
    wts = wts / np.sum(wts, axis=1)[:, np.newaxis]
    # format the tar get
    y = np.zeros((ab.shape[0], nb_q))
    idx_pts = np.arange(ab.shape[0])[:, np.newaxis]
    y[idx_pts, idx_neigh] = wts
    y = y.reshape(h, w, nb_q)
    return y

class CIE_Iamges(Dataset):
    nb_neighbors = 5
    
    def __init__(self,
                 imgs_list:list,
                 soft_encode:bool = False
                 ):
        """
        Custom dataset class for images converted to LAB space color.

        Args:
            imgs_list (list): A list of image paths.
        """
        super(CIE_Iamges, self).__init__()
        
        self.imgs_list   = imgs_list
        self.soft_encode = soft_encode
        
        if self.soft_encode == True:
            q_ab = np.load("./pts_in_hull.npy")
            self.nb_q = q_ab.shape[0]
            # Fit a NN to q_ab
            self.nn_finder = nn.NearestNeighbors(n_neighbors=self.nb_neighbors, algorithm='ball_tree').fit(q_ab)

    def __getitem__(self,
                    index:int):
        """
        Retrieves an item from the dataset at the specified index.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing the preprocessed L channel of the
            image as a Torch Tensor and the original image as a PIL Image.
        """
        image_path = self.imgs_list[index]
        image = Image.open(image_path).resize((utils.img_height,utils.img_width)).convert('RGB')

        Resized_RGB_LAb_Lumination,Resized_RGB_LAb_ab = preprocess_img(np. array(image))
        Resized_RGB_LAb_Lumination_Tensor   = transforms.ToTensor()(Resized_RGB_LAb_Lumination).to(torch.float)
        Resized_RGB_LAb_ab_Tensor           = transforms.ToTensor()(Resized_RGB_LAb_ab        ).to(torch.float)
        
        
        if self.soft_encode == True:
            image_64_64 = Image.open(image_path).resize((64,64)).convert('RGB')
            _,Resized_RGB_LAb_ab_64_64 = preprocess_img(np. array(image_64_64))
            Resized_RGB_LAb_ab = get_soft_encoding(Resized_RGB_LAb_ab_64_64, self.nn_finder, self.nb_q)
            
            Resized_RGB_LAb_Lumination_Tensor   = transforms.ToTensor()(Resized_RGB_LAb_Lumination).to(torch.float)
            Resized_RGB_LAb_ab_Tensor           = transforms.ToTensor()(Resized_RGB_LAb_ab        ).to(torch.float)
            
            return Resized_RGB_LAb_Lumination_Tensor,Resized_RGB_LAb_ab_Tensor
            
        
        return Resized_RGB_LAb_Lumination_Tensor,Resized_RGB_LAb_ab_Tensor

    
    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
            int: The total number of items in the dataset.
        """
        return len(self.imgs_list)
    
#######################################################################
#######################################################################
from torch.utils.data       import DataLoader
   
imagesadresses = crawler_hand_picked()
full_dataset = CIE_Iamges(imagesadresses,soft_encode=utils.soft_encode)

train_size = int(0.80 * len(imagesadresses))
Valid_size = len(imagesadresses) - train_size

train_dataset, test_dataset  = torch.utils.data.random_split(full_dataset,  [train_size, Valid_size])

train_dataloader = DataLoader(train_dataset, batch_size=80, shuffle=True,num_workers=2)
test__dataloader = DataLoader(test_dataset, batch_size=20, shuffle=True,num_workers=2)