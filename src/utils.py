import numpy as np
import pandas as pd

"""
https://stats.stackexchange.com/questions/104771/latent-dirichlet-allocation-in-pymc
https://discourse.pymc.io/t/lda-implementation-with-pymc3/1515
http://alfan-farizki.blogspot.com/2015/07/pymc-tutorial-3-latent-dirichlet.html
https://stackoverflow.com/questions/25085316/implementing-latent-dirichlet-allocation-lda-with-pymc
https://github.com/napsternxg/ipython-notebooks/blob/master/PyMC_LDA.ipynb
"""

documents = ["I had a peanuts butter sandwich for breakfast.",
             "I like to eat almonds, peanuts and walnuts.",
             "My neighbor got a little dog yesterday.",
             "Cats and dogs are mortal enemies.",
             "You mustn’t feed peanuts to your dog."]
