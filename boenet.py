import torch.nn as nn
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data.dataset import ConcatDataset
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests


### API KEYShsl"
HUG_API_KEY = "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"
os.environ['HF_TOKEN'] = HUG_API_KEY
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'ls__fe633ef5a71843baa5d07db00d83cd68'
os.environ['PINECONE_API_KEY'] = "db004a52-8d38-49e6-8731-0f0a562d10b1"


# Request to create embeddings
model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {HUG_API_KEY}"}


class BOENet(nn.Module):
    def __init__(self, d_model :int , boe_labels : int) -> None:
       super().__init__()
       self.d_model = d_model # (384)
       self.boe_labels = boe_labels # (?)
       
       # Arquitecture 
       # (batch, 384)
       self.l_1 = nn.Linear(in_features= self.d_model, out_features = 700, bias = True)
       self.tan_h = nn.Tanh()
       self.drop_1 = nn.Dropout(p = 0.2)
       self.l_2 = nn.Linear(in_features= 700, out_features = 1200, bias = True)
       self.Relu = nn.ReLU()
       self.drop_2 = nn.Dropout(p = 0.3)
       # (batch, 10)
       self.l_3 = nn.Linear(in_features=  1200 , out_features = self.boe_labels, bias = True)
       
    def forward(self,x):
        h = self.drop_1(self.tan_h(self.l_1(x)))
        h = self.drop_2(self.Relu(self.l_2(h)))
        return self.l_3(h)
    
    @staticmethod
    def LossFactory():
        return nn.CrossEntropyLoss()
    
    @staticmethod
    def OptimizerFactory(model, lr : float = 0.001 , betas : tuple = (0.9, 0.999), eps: float =1e-08):
        return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)

    
# Dataset
class BOEData(Dataset):
    def __init__(self, path: str) :
        super().__init__()
        
        # raw data in form of df
        self.data = pd.read_csv( filepath_or_buffer = path, delimiter = ',')
        
        # Create samples and target codify labels to train net
        self._map_labels()
        self.x = torch.tensor(self.get_embeddings(self.data.loc[:,'text'].to_list()))
        self.y = torch.tensor(self.data.loc[:,'maped_labels'].values)
        
    def __getitem__(self, index):
        return self.x[index] ,self.y[index]
    def __len__(self):
        return self.data.shape[0]

    def _map_labels(self):
        mapping = {}
        for i,l_i in enumerate(self.data['label'].unique()):
            mapping[l_i] = i + 1
        self.data['maped_labels'] = self.data['label'].map(mapping)
         
    def get_embeddings(self,texts):
        response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
        self.embeddings = response.json()
        return response.json()

            
        



# MODULE CLASS DOCU:
"""nn.Module :
Base class for all neural network modules.

Your models should also subclass this class.

Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes: 
self.sub_module = nn.Linear(...)"""

# Iterate along all de modules inside a network class or model class. Notice that LinearRegression module has inside a linear layer "module" or only linear layer
# note that the atribute name : self.linear will define the string "linear" to refer to that layer inside a module
# this will be useful when ,inside a class module, there are several layers


if __name__ == '__main__':
    
    # instanciar capa y crear modelo
    model = BOENet(d_model  = 384 , boe_labels = 10)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Cuda available: ",torch.cuda.is_available())
    model.to(device)
    
    # loss class
    loss = BOENet.LossFactory() 
    
    # crea optimizer para prescendir de actualizacion de pesos a mano
    optimizer = BOENet.OptimizerFactory(model) 
    
    print("\n")

    for m in model.modules():
        #print(f"\nModule : {m}")
        pass

    # iterate along all the parameters inside a module ( a module can have a lot of layers with their parameters)
    for p in model.named_parameters(prefix='', recurse=True, remove_duplicate=True):
        #print(f"\nParameter : {p}")
        pass
        
    docs = [
        'hola que tal esto es un ejemplo',
        'hola ejemplo 2'
    ]
    
    embedding = [
        np.random.rand(384),
        np.random.rand(384)
        
    ]
    labels = [
        'label 1',
        'label 2'
    ]
    data = {
        'text':docs ,
        'label': labels,
    }
    df_data = pd.DataFrame(data = data)
    df_data.to_csv(path_or_buf='./Data_BOE/filename.csv')
    print(df_data.head(10))
    
    data = BOEData(path = './Data_BOE/filename.csv')
    print(len(data))
    print((data.x.shape))
    print((data.x[0,:].shape))
    
    BATCH_SIZE = 2
    train_loader = torch.utils.data.DataLoader(dataset = data, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = data, batch_size = BATCH_SIZE, shuffle = False)
    print(data.data["text"])
    print(data.data["label"])
    for i,(x,y)in enumerate(train_loader):
        print(i, x.shape, y.shape)
        print(y)
    
    

    
    