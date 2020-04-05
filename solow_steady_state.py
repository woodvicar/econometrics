"1|IMPORT PACKAGES"
import numpy as np               # Package for scientific computing with Python
import matplotlib.pyplot as plt  # Matplotlib is a 2D plotting library

"2|DEFINE PARAMETERS AND ARRAYS|"
# Parameters
K_size = 101                     # Model domain
A = 1                            # Total Factor Productivity
alpha = 0.50                     # Output elasticity of capital
delta = 0.03                     # Depreciation rate
s1 = 0.35                        # Savings rate before the shock
s2 = 0.45                        # Savings rate after the shock
n  = 0.02                        # Population growth rate
# Arrays
k  = np.arange(K_size)           # Create array of k

"3|DEFINE FUNCTIONS"
def output(k):                   # Cobb-Douglas per capita function
    y = A * (k)**(alpha)
    return y

y  = output(k)                   # Production function
d  = delta*k                     # Depreciation
i1 = s1*y                        # Investment before the shock
i2 = s2*y                        # Investment after the shock
d_and_i = (delta + n)*k          # Breack-even

"4|CALCULATE STEADY-STATE VALUES"
k_star1 = (s1/(n+delta)*A)**(1/(1-alpha))
k_star2 = (s2/(n+delta)*A)**(1/(1-alpha))
y_star1 = A*(k_star1**alpha)
y_star2 = A*(k_star2**alpha)
i_star1 = s1*y_star1
i_star2 = s2*y_star2
c_star1 = y_star1 - i_star1
c_star2 = y_star2 - i_star2
d_star1 = delta*k_star1
d_star2 = delta*k_star2



"6|SAVINGS RATE: ONE-PERIOD SHOCK"
T = 200                 # Number of periods
t_shock = 10            # Period when shock happens
time = np.arange(T)     # Create array of time
s = np.zeros(T)         # Create array of s
y = np.zeros(T)         # Create array of y
k = np.zeros(T)         # Create array of k
i = np.zeros(T)         # Create array of i
c = np.zeros(T)         # Create array of c

y[0] = y_star1          # Set initial value of y
k[0] = k_star1          # Set initial value of k
i[0] = i_star1          # Set initial value of i
c[0] = c_star1          # Set initial value of c

s = np.zeros(T)
s[0:T] = s1             # Array of savings rate
s[t_shock] = s2         # Shock to savings rate

for j in range(1, T):
    k[j] = k[j-1] + i[j-1] - (n + delta)*k[j-1]
    y[j] = A*k[j]**alpha
    i[j] = s[j]*y[j]
    c[j] = -(y[j] - i[j])
    
### Plot effect on variables
ticks = [""]*T                                  # Create tick labels
ticks[t_shock] = 'Shock'                        # Create label "shock" 

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
fig.subplots_adjust(hspace=0)                   # Plots be next to each other
ax1.set(title=" SHOCK TO SAVINGS RATE")
ax1.plot(time, k, "k-", alpha = 0.7)
ax1.axvline(x = t_shock, color="k", ls = ':', alpha = 0.6)
ax1.yaxis.set_major_locator(plt.NullLocator())  # Hide ticks
ax1.text(150, 49.1, 'Capital: '+r'$k$')

ax2.plot(time, y, "b-", alpha = 0.7)
ax2.axvline(x = t_shock, color="k", ls = ':', alpha = 0.6)
ax2.yaxis.set_major_locator(plt.NullLocator())  # Hide ticks
ax2.text(150, 7.01, 'Output: '+ r'$y=f(k)$', color = "b")

ax3.plot(time, i, "g-", alpha = 0.7)
ax3.plot(time, c, "r-", alpha = 0.7)
ax3.axvline(x = t_shock, color="k", ls = ':', alpha = 0.6)
ax3.yaxis.set_major_locator(plt.NullLocator())  # Hide ticks
ax3.xaxis.set_major_locator(plt.NullLocator())  # Hide ticks
ax3.text(150, 4.2, 'Consumption: '+r'$c = (1-s)y$', color = "r")
ax3.text(150, 2.7, 'Investment: '+r'$i = sy$'     , color = "g")
plt.xticks(time, ticks)                         # Use user-defined ticks
plt.xlabel('Time')
plt.tick_params(axis='both', which='both', bottom=False, top=False,
                labelbottom=True, left=False, right=False, labelleft=True)
                                                # Hide tick marks

"8|SAVINGS RATE: PERMANENT SHOCK"
time = np.arange(T)     # Create array of time
         # Create array of s
y = np.zeros(T)         # Create array of y
        # Create array of i
c = np.zeros(T)         # Create array of c

y[0] = y_star1          # Set initial value of y
k[0] = k_star1          # Set initial value of k
i[0] = i_star1          # Set initial value of i
c[0] = c_star1          # Set initial value of c

s = np.zeros(T)
s[0:t_shock] = s1       # Array of savings rate
s[t_shock:T] = s2       # Shock to savings rate

for j in range(1, T):
    k[j] = k[j-1] + i[j-1] - (n + delta)*k[j-1]
    y[j] = A*k[j]**alpha
    i[j] = s[j]*y[j]
    c[j] = -(y[j] + i[j])
    
### Plot effect on variables
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
fig.subplots_adjust(hspace=0)                    # Plots be next to each other
ax1.set(title="INCREASE IN THE SAVINGS RATE WHILE ABOVE GOLDEN LEVEL")
ax1.plot(time, y, "b-", alpha = 0.7)
ax1.axvline(x = t_shock, color="k", ls = ':', alpha = 0.6)
ax1.yaxis.set_major_locator(plt.NullLocator())   # Hide ticks
ax1.text(150, 8.7, 'Output: '+ r'$y=f(k)$', color = "b")

ax2.plot(time, c, "r-", alpha = 0.7)
ax2.axvline(x = t_shock, color="k", ls = ':', alpha = 0.6)
ax2.yaxis.set_major_locator(plt.NullLocator())   # Hide ticks
ax2.xaxis.set_major_locator(plt.NullLocator())   # Hide ticks
ax2.text(150, 3.1, 'Consumption: '+r'$c = (1-s)y$', color = "r")
plt.xticks(time, ticks)                          # Use user-defined ticks
plt.xlabel('Time')
plt.tick_params(axis='both', which='both', bottom=False, top=False,
                labelbottom=True, left=False, right=False, labelleft=True)
                                                # Hide tick marks                                              # Hide tick marks
plt.show()

