from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

n = 100
p = 0.16

my_binom = binom(n,p)

n_wins = np.arange(0,40,1)
plt.figure()
plt.plot(n_wins,my_binom.pmf(n_wins), "o")
plt.show

print(my_binom.sf(15))
print(my_binom.cdf(24))

#vectorise inputs to get a result 
import numpy as np
from scipy.stats import norm
def f(x):
    return np.int(x)
f2 = np.vectorize(f)
x = np.arange(1, 15.1, 0.1)
plt.plot(x, f2(x))
plt.show()

