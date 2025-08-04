# Leveraging covariance representations with Riemannian ResNet for satellite image time series classification

This code is supporting by a conference paper that will  be published in 2025 IEEE International Geoscience and Remote Sensing Symposium proceedings:

@inproceedings{Zakir2025Leveraging,
    Title = {Leveraging covariance representations with Riemannian ResNet for satellite image time series classification},
    Author = {Zakir, Khizer and Pelletier, Charlotte and Chapel, Laetitia and Courty, Nicolas},
    Booktitle = {IEEE International Geoscience and Remote Sensing Symposium (IGARSS) 2025},
    note = {Accepted for an oral presentation}
}


## Prerequisites

This codes relies on Python 3.12  and PyTorch 2.5. It also requires the download of the `BreizhCrops` dataset. To obtain the results of the paper, you need to run the following code that will install a virtual env:  
`python3 -m venv lernenv` # creating a virtual environment `lernenv`  
`source lernenv/bin/activate` # activate the virual environment  
`pip install -r requirements.txt` # install the required modules  
`patch -p1 -d lernenv/lib/python3.12/site-packages < ./patches/geoopt_scalar_wolfe1.patch` # patch geoopt to make it compatible with the newest version of scipy  

## Running

The results can be reproduced using `./run_all.sh`. The script can also be modified to use different configurations.

## Contributors
 - [Khizer Zakir](https://khizerzakir.github.io/)
 - [Dr. Charlotte Pelletier](https://sites.google.com/site/charpelletier)
 - [Professor Laetitia Chapel](https://people.irisa.fr/Laetitia.Chapel/)
 - [Professor Nicolas Courty](https://ncourty.github.io/)

