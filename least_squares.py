from numpy import loadtxt,power,linspace
from matplotlib.pyplot import plot,show,xlabel,ylabel

data=loadtxt("milikan.txt")

voltage = data[:,1]
frequency = data[:,0]
N=len(frequency)
#beregning af vægte
Ex = 1/N*sum(frequency)
Ey = 1/N*sum(voltage)
Exx = 1/N*sum(power(frequency,2))
Exy = 1/N*sum(frequency*voltage)

a = (Exy-Ex*Ey)/(Exx-power(Ex,2))
b = (Exx*Ey-Ex*Exy)/(Exx-Ex**2)
x = linspace(frequency[0],frequency[N-1],100)
y=a*x+b

plot(frequency,voltage,'bo')
plot(x,y,'g')
xlabel("Frequency in Herz")
ylabel("Voltage in Volts")
show()
