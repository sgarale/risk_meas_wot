## Risk measures based on weak optimal transport
---------------------
This repository contains the implementation of the numerical examples in ["Risk measures based on weak optimal transport"]()

### Requirements
The file [requirements.yml](requirements.yml) contains minimal requirements for the environment necessary to run the notebooks of this repository.

### Notebooks

Here, a list of the notebooks used to produce the plots of the paper:
- [earthquake.ipynb](earthquake.ipynp) simulates the earthquake model of Section 5.1 (Figures 1 - 2).
- [martingale_1d.ipynb](martingale_1d.ipynb) reproduces the C-transform of the bull spread option appearing in Section 5.2 (Figure 3a).
- [bull_spread_bounds.ipynb](bull_spread_bounds.ipynb) reproduces the arbitrage-free bounds for the bull spread option of Section 5.2 (Figure 3b).
- [martingale_2d.ipynb](martingale_2d.ipynb) studies the Max Call option on two assets of Section 5.2.2 (Figure 4).
- [higher_dimensios.ipynb](higher_dimensions.ipynb) computes the upper bound on the price of options written on an increasing number of assets. It reproduces the plots of Figure 5. It is possible to choose the option at the beginning of the notebook.

Execution time may vary depending on the machine.

#### Version 1.0

### License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details
