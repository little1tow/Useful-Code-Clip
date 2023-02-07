"""
@Author: zhkun
@Time:  2022-03-27 17:05
@File: text
@Description: leveraging plt.annotate() to point out specific point
@Something to attention
"""

import matplotlib.pyplot as plt
import numpy as np

# using plt.annotate() to point out the specific point
fig = plt.figure(figsize=(8,6))
X = list(range(10))
plt.plot(X, np.exp(X))
plt.title('Annotating exponential plot using plt.annotate()')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.annotate('Point1', xy=(6, 400),
	         arrowprops=dict(arrowstyle='->'),
	         xytext=(4, 600))

plt.annotate('Point2', xy=(7, 1150),
	         arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=-.2'),
	         xytext=(4.5, 2000))

plt.annotate('Point3', xy=(8, 3000),
	         arrowprops=dict(arrowstyle='->', connectionstyle='angle, angleA=90,angleB=0'),
	         xytext=(8.5, 2200))

plt.show()


# using box plot to better deonstrate the results (unkonw application)

import numpy as np
import matplotlib.pyplot as plt

def list_generator(mean, dis, number):
    return np.random.normal(mean, dis * dis, number)  # normal


# generate example data
girl20 = list_generator(1000, 29.2, 70)
boy20 = list_generator(800, 11.5, 80)
girl30 = list_generator(3000, 25.1056, 90)
boy30 = list_generator(1000, 19.0756, 100)

data = [girl20, boy20, girl30, boy30]
label = ['Girl 20', 'Boy 20', 'Girl 30', 'Boy 30']
plt.figure(figsize=(10, 5))
plt.title('example of boxplot', fontsize=23)

# basic usage
plt.boxplot(data, labels=label) # grid is false means background line is invisible
plt.show()


# horizen usage
plt.boxplot(data, labels=label, vert=False, showmeans=True) # vert=False -> HORIZEN boxplot, 
                                                                        # showmeans=True -> Show the mean value of all the bars

plt.show()

# using different shape of boxplot
plt.boxplot(data, notch=True, sym='*')
plt.show()

