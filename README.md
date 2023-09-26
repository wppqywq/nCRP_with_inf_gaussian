This repo has two parts of work showing my study of ML theory in 2023 summer.


## 1. Joint the nested chinese restuarent process with the stochastic beam search ##

    - [Stochastic_Beam_Search.ipynb](Stochastic_Beam_Search.ipynb): implement the stochastic beam search based on paper (https://arxiv.org/abs/1903.06059), used The Gumbel-Top-k Trick for sampling.
  
    - [nCRP_with_beam_search.ipynb](nCRP_with_beam_search.ipynb): impelement the nested chinese resturant process and apply the beam search, based on the paper (https://proceedings.neurips.cc/paper/2003/file/7b41bfa5085806dfa24b8c9de0ce567f-Paper.pdf).


## 2. Infinite Gaussian Mixture Model ##

Modified code from (https://github.com/Vrroom/infiniteGMM)

Implementation and explanation of the paper [The Infinite Gaussian Mixture Model](https://www.seas.harvard.edu/courses/cs281/papers/rasmussen-1999a.pdf).

    - [sampler.py](sampler.py):  Implements adaptive rejection sampling and gibbs sampling.

    - [distributions.py](distributions.py): Contains representation of a piecewise exponential distribution.

    - [Infinite Gaussian Mixture Model.ipynb](Infinite Gaussian Mixture Model.ipynb): Implementation of the method along with my running commentary.
  
    - [inf_nCRP.ipynb](inf_nCRP.ipynb): further modification that to be complete...
