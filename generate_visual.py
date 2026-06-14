import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_pso_sphere():
    # Define the Sphere function
    def sphere(x, y):
        return x**2 + y**2

    # Create a grid of points for the surface
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = sphere(X, Y)

    # Simulate some PSO particles moving towards the minimum (0,0)
    # We will plot particles at an initial state and an intermediate state
    np.random.seed(42)
    
    # Initial scattered particles
    particles_x_init = np.random.uniform(-8, 8, 15)
    particles_y_init = np.random.uniform(-8, 8, 15)
    particles_z_init = sphere(particles_x_init, particles_y_init)
    
    # Intermediate state (closer to the minimum)
    particles_x_mid = particles_x_init * np.random.uniform(0.2, 0.6, 15)
    particles_y_mid = particles_y_init * np.random.uniform(0.2, 0.6, 15)
    # Add some noise
    particles_x_mid += np.random.normal(0, 0.5, 15)
    particles_y_mid += np.random.normal(0, 0.5, 15)
    particles_z_mid = sphere(particles_x_mid, particles_y_mid)

    # Converged point (global best)
    best_x, best_y = 0, 0
    best_z = sphere(best_x, best_y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

    # Plot the particles
    ax.scatter(particles_x_init, particles_y_init, particles_z_init, color='red', s=40, label='Initial Swarm')
    ax.scatter(particles_x_mid, particles_y_mid, particles_z_mid, color='orange', s=40, label='Intermediate Swarm')
    ax.scatter([best_x], [best_y], [best_z], color='blue', s=100, marker='*', label='Global Minimum')

    # Add lines showing movement for a few particles
    for i in range(5):
        ax.plot([particles_x_init[i], particles_x_mid[i]], 
                [particles_y_init[i], particles_y_mid[i]], 
                [particles_z_init[i], particles_z_mid[i]], color='gray', linestyle='--', alpha=0.7)
        ax.plot([particles_x_mid[i], best_x], 
                [particles_y_mid[i], best_y], 
                [particles_z_mid[i], best_z], color='gray', linestyle=':', alpha=0.7)

    ax.set_title("PSO Minimizing the Sphere Function")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("f(X, Y)")
    ax.legend()
    
    # Make sure assets directory exists
    os.makedirs('assets', exist_ok=True)
    
    # Save the figure
    plt.savefig('assets/pso_sphere.png', dpi=300, bbox_inches='tight')
    print("Figure saved to assets/pso_sphere.png")

if __name__ == "__main__":
    plot_pso_sphere()
