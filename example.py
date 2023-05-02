import math

ex = [
    [1,2],
    [3,4],
    [5,6]
]

x,y = zip(*ex)

def calc_accel(beta, b1, b2, b3, mu, r, alpha, e_r, n):
  ''' Calculate the acceleration of the solar sail
  '''
  alpha_deg = alpha * 180 / math.pi
  accel = []
  return [(beta/(b1+b2+b3))*(mu/r**2)*math.cos(alpha_deg)*(b1*e_r[i] + (b2*math.cos(alpha_deg))*n[i]) for i in range(len(n))]

print(calc_accel(1,1,1,1,1,1,1,[0.6,0.6],[0.6,0.6]))