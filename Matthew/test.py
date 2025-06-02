import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 1.0        # wave speed (you can change)
t = 0.5        # fixed time (you can change)

# x values
x = np.linspace(-10, 10, 400)

# Function u(x,t)
u = 4 * c / (c**2 * (x - c * t)**2 + 1)

# Plot
plt.plot(x, u, label=f't={t}, c={c}')
plt.xlabel('x')
plt.ylabel('u(x,t)')
plt.title('Plot of u(x,t) = 4c / (c^2 (x - ct)^2 + 1)')
plt.legend()
plt.grid(True)
plt.show()