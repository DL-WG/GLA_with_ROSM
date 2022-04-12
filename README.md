# GLA_with_ROSM
In this project, we introduce the Generalised Latent Assimilation (GLA) approach which can be used in Machine learning surrogate modelling for dynamical systems 
The reportory including python scripts for reduced-order-modelling (POD, 1D CAE and POD AE), LSTM surrogate modelling (inside the latent space with preprocessing) and GLA (with non-linear high dimensional transformation operators).

If you are interested in applying GLA for a inverse problem with an initial guess, please refer to the file GLA_simple_ex.py in the GLA repotory. The ROM_LSTM_GLA.py contains the implementation of GLA with a reduced-order surrogate model (with either POD or POD AE + LSTM).


Due to the large volume, we provide only 5 time steps of oil concentration data (The CFD data for one simulation (Um = 0.52)) in the two-phase flow application. To read and reshape openfoam data, please refer to the python script load_data.py in ROM repotory. The full CFD dataset is available upon reasonalble request to sibo.cheng@imperial.ac.uk

citation:

@article{cheng2022generalised,
  title={Generalised Latent Assimilation in Heterogeneous Reduced Spaces with Machine Learning Surrogate Models},
  author={Cheng, Sibo and Chen, Jianhua and Anastasiou, Charitos and Angeli, Panagiota and Matar, Omar K and Guo, Yi-Ke and Pain, Christopher C and Arcucci, Rossella},
  journal={arXiv preprint arXiv:2204.03497},
  year={2022}
}
