# BestSecretApp

## Steps to setup the Best Secretlit Stream App

**Step 1**

**First Setup your enviroment in your local machine and install all the relevant packages**

conda create --name best_secret

conda activate best_secret

conda install python=3.10.12

conda install matplotlib

conda install tensorflow

conda install conda-forge::streamlit -y

conda install -c conda-forge opencv

**Step 2**

**In your command line terminal go to the root folder where you want to save the streamlit GitHub project**

cd root_folder_path

**Clone the project from GitHub**

git clone git@github.com:anurag-chowdhury1975/BestSecretApp.git

Once the cloning is complete you should see the following directory structure created:

* BestSecretApp
    * notebooks
    * src

**Step 3**

**The datasets and models have not been added to the GitHub project because of privacy issues. You will need to first create the following folders before downloading the datasets and models to your local drive from Best Secrets Google Drive folder:**

cd BestSecretApp

mkdir data

mkdir models

After this your project folder structure should look like this:

* BestSecretApp
    * data
    * models
    * notebooks
    * src

**Step 4**

**Download ONLY the test datasets for each product category from Best Secrets Google Drive folder https://drive.google.com/drive/folders/1_difVXO-_N1iMFzxMP2IFeD7e8tUhRBV, to your local project folder.**

Your local project folder structure should look like this after you download the test datasets:
* BestSecretApp
    * data
        * bag
            * test_dataset
        * clothes
            * test_dataset
        * schuhe
            * test_dataset
        * waesche
            * test_dataset

**Step 5**

**Next you need to download the latest best models for each product from the GoogleDrive folder https://drive.google.com/drive/folders/1JUgLVKtQinZkC79GIsnu6YwbAESWpntX, into your local project folder**

Your local project folder structure should look like this after you download the models:

(NOTE: the model file names listed below are the best models we have as of 14/02/2024, if we get better models later you need to add these to the local project as well):
* BestSecretApp
    * models
        * bag_resnet50_model_ft_all_93%.h5
        * clothes_resnet50_func_model_97%.h5
        * models/schuhe_resnet50_model_ft_all_94%.h5
        * models/waesch_funcResnet_model_94%.h5

**Step 6**

**You should be all set to run the BestSecret streamlit app from your local machine now using the following command:**

(NOTE: make sure you are in the BestSecretApp folder before you run the command below.)

streamlit run src/bestsecret_app.py