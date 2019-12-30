# Source code of Auria Kathi - The first AI Poet Artist


## Data Sources and trained models

For setting up Azure, I've uploaded all the required files to Azure Machine Learning Datastores. Whenever the data is required, it is referred from the respective Datastore. 

If you would like to set things up locally, the following links are useful

**Haikus for Language Model**

- Cleaned haiku dataset: https://bit.ly/2MHlFMu

**AttnGAN**

- Image Encoder - https://bit.ly/2MGBnaH 
- Text Encoder - https://bit.ly/2ZA72zX 
- AttnGAN Model - https://bit.ly/2F34vEZ
- COCO dataset - https://bit.ly/367Ttdn  

I've written a convenient clone of AttnGAN [here](https://github.com/sleebapaul/attnGAN)

Refer the repo for setting up all these requirements using a single shell script. This is the easier way, you may try setting things up from the official repo as well. 

**GPT-2 finetuning** 

- GPT-2 finetuning - https://github.com/minimaxir/gpt-2-simple  

**FastPhotoStyle**

- FastPhotoStyle - https://github.com/NVIDIA/FastPhotoStyle


## Folders

**AttnGAN**

Code of a convenient clone of AttnGAN by MSFT Research. AttnGAN original repo contains the training and testing code. But this folder is solely intended to use for generating images from text using MSCOCO pretrained weights.
- Original paper : http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf
- Original repo : https://github.com/taoxugit/AttnGAN

**Dockerfiles**

Dockerfiles and requirements required for building various docker containers needed to build Auria's pipeline steps.

**GPT_2_Finetuning**

Code of convenient clone of finetuning the `GPT-2` model using 117M pretrained model and then generate text from it. 

- Original paper: https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
- Original repo: https://github.com/minimaxir/gpt-2-simple

**Notebooks**

Different `Jupyter Notebooks` used to build training, testing and building pipelines of Auria. 

**NVIDIAFastPhotoStyle**

Code of convenient clone of FastPhotoStyle by NVIDIA. 

- Original paper: https://arxiv.org/abs/1802.06474
- Original repo: https://github.com/NVIDIA/FastPhotoStyle


## Azure ML Pipelines

1. You'll need an Azure ML Pipelines subscription and its configuration JSON file to use the platform. You can refer the notebooks to know the usage of these config files. 

2. Also install the `azure_requirements.txt`.

```pip3 install -r azure_requirements.txt```

## The Medium Articles for overview

1. [Auria Kathi — An artist in the clouds](https://towardsdatascience.com/auriakathi-596dfb8710d6)  
2. [Auria Kathi Powered by Microsoft Azure Machine Learning Pipelines](https://towardsdatascience.com/auria-kathi-powered-by-microsoft-azure-machine-learning-pipelines-385de55de062)

