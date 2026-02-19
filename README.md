# EP.ROBOTICA
A.	Sin interacción de consola
1. 
import numpy as np

# Vectores previamente inicializados
v1 = np.array([2, 4, 6])
v2 = np.array([1, 3, 5])

# Suma
suma = v1 + v2

# Resta
resta = v1 - v2

# Producto punto
producto_punto = np.dot(v1, v2)

# Producto cruz
producto_cruz = np.cross(v1, v2)

# División elemento a elemento
division = v1 / v2

print("Suma:", suma)
print("Resta:", resta)
print("Producto Punto:

2. 
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Suma:\n", A + B)
print("Resta:\n", A - B)
print("Multiplicación:\n", np.dot(A, B))
print("Producto elemento a elemento:\n", A * B)
print("División elemento a elemento:\n", A / B)

", producto_punto)
print("Producto Cruz:", producto_cruz)
print("División:", division)

3.
import math

x = 3
y = 4
z = 5

# Cilíndricas
r = math.sqrt(x**2 + y**2)
theta = math.atan2(y, x)
z_cil = z

# Esféricas
rho = math.sqrt(x**2 + y**2 + z**2)
phi = math.acos(z / rho)

print("Cilíndricas (r,θ,z):", r, theta, z_cil)
print("Esféricas (ρ,θ,φ):", rho, theta, phi)

4.
R0 = 100
A = 3.9083e-3
B = -5.775e-7

T = 50  # Temperatura ejemplo

R = R0 * (1 + A*T + B*T**2)

print("Resistencia PT100:", R, "Ohm")

5.
import numpy as np
import math

def rot_x(theta):
    return np.array([
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)]
    ])

def rot_y(theta):
    return np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]
    ])

def rot_z(theta):
    return np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ])

print(rot_x(math.radians(30)))
print(rot_y(math.radians(30)))
print(rot_z(math.radians(30)))

6. 
import math

P = 600000  # Pa
diametro = 0.05  # m
vástago = 0.02  # m

A_avance = math.pi * (diametro/2)**2
A_retroceso = math.pi * ((diametro/2)**2 - (vástago/2)**2)

F_avance = P * A_avance
F_retroceso = P * A_retroceso

print("Fuerza Avance:", F_avance, "N")
print("Fuerza Retroceso:", F_retroceso, "N")

B. Con interacción de consola (fprintf o disp) y teclado (input)
1.
V = float(input("Ingrese Voltaje: "))
I = float(input("Ingrese Corriente: "))

P = V * I

print("Potencia:", P, "W")

2.
import random

cantidad = int(input("Cantidad: "))
minimo = int(input("Valor mínimo: "))
maximo = int(input("Valor máximo: "))

for i in range(cantidad):
    print(random.randint(minimo, maximo))

3.
import math

print("1.Prism 2.Piramide 3.Cono truncado 4.Cilindro")
op = int(input("Seleccione: "))

if op == 1:
    A = float(input("Area base: "))
    h = float(input("Altura: "))
    print("Volumen:", A*h)

elif op == 2:
    A = float(input("Area base: "))
    h = float(input("Altura: "))
    print("Volumen:", (A*h)/3)

elif op == 3:
    R = float(input("Radio mayor: "))
    r = float(input("Radio menor: "))
    h = float(input("Altura: "))
    print("Volumen:", (math.pi*h/3)*(R**2 + r**2 + R*r))

elif op == 4:
    r = float(input("Radio: "))
    h = float(input("Altura: "))
    print("Volumen:", math.pi*r**2*h)

4.
print("1.Cartesiano 2.Cilindrico 3.Esferico")
op = int(input("Seleccione robot: "))

if op == 1:
    print("Robot Cartesiano - 3 articulaciones prismáticas")
elif op == 2:
    print("Robot Cilíndrico - 1 rotacional y 2 prismáticas")
elif op == 3:
    print("Robot Esférico - 2 rotacionales y 1 prismática")

5.
respuesta = ""

while respuesta.lower() != "no":
    respuesta = input("Desea continuar Si/No? ")

C. Uso de las funciones para graficar
1.
import numpy as np
import matplotlib.pyplot as plt

R0 = 100
A = 3.9083e-3
B = -5.775e-7

T = np.linspace(-200,200,400)
R = R0*(1 + A*T + B*T**2)

plt.plot(T,R)
plt.xlabel("Temperatura °C")
plt.ylabel("Resistencia Ω")
plt.title("Sensor PT100")
plt.grid()
plt.show()

2. 
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

num = [1]
den = [1, 2, 1]

sistema = signal.TransferFunction(num, den)
t, y = signal.step(sistema)

plt.plot(t,y)
plt.title("Respuesta al Escalón")
plt.grid()
plt.show()

3.
import numpy as np
import matplotlib.pyplot as plt

V = float(input("Voltaje: "))
R = float(input("Resistencia: "))
C = float(input("Capacitancia uF: ")) * 1e-6

t = np.linspace(0,5*R*C,500)

Vc = V*(1 - np.exp(-t/(R*C)))

plt.plot(t,Vc)
plt.title("Carga RC")
plt.grid()
plt.show()

4.
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = float(input("X: "))
y = float(input("Y: "))
z = float(input("Z: "))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(0,0,0,x,y,z)

plt.show()

5.
import matplotlib.pyplot as plt

plt.text(0.2,0.5,"ELIAS", fontsize=30)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

6.
import numpy as np
import matplotlib.pyplot as plt

theta = np.linspace(0,2*np.pi,300)

# Chevrolet 
x1 = 2*np.cos(theta)
y1 = np.sin(theta)

# Hyundai 
x2 = 3*np.cos(theta)
y2 = 1.5*np.sin(theta)

plt.plot(x1,y1)
plt.plot(x2,y2)
plt.axis("equal")
plt.title("Contornos aproximados")
plt.grid()




plt.show()

