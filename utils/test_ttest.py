from scipy.stats import ttest_ind

x = [1,2,3,4,5,6,7,8,9,11,22,33,44,55,66,77,88,99]
y = [99,88,77,66,55,44,33,22,11,9,8,7,6,5,4,3,2,1]

print(ttest_ind(x,y))