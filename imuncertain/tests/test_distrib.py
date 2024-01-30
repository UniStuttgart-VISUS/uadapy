import numpy as np
import scipy as sp
import scipy.stats as st
import distribution



def test_distrib_class():

    model1D = {
        st.alpha(1.0),
        st.anglit(),
        st.arcsine(),
        st.argus(1.0),
        st.beta(1.0, 1.0),
        st.betaprime(1.0, 1.0),
        st.bradford(1.0),
        st.burr(1.0, 1.0),
        st.burr12(1.0, 1.0),
        st.cauchy()
    }


    