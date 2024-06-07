"""Script to compute and visualize a restricted 2 body problem.

That is, a small object orbiting a planet.

Source:
https://www.youtube.com/watch?v=7JY44m6eemo
https://en.wikipedia.org/wiki/Standard_gravitational_parameter#:~:text=In%20celestial%20mechanics%2C%20the%20standard,mass%20M%20of%20the%20bodies.

"""
from scipy.integrate import ode
from scipy.constants import gravitational_constant
import matplotlib.pyplot as plt
import numpy as np


EARTH_RADIUS = 6371.0  # km
EARTH_MASS = 5.972 * (10**24)  # kg
EARTH_MU = gravitational_constant * EARTH_MASS  # m ^ 3 / s^2
EARTH_MU = EARTH_MU / (1000**3)  # Redefine to km^3 / s^2


def plot(trajectory: np.ndarray):
    """Plot the trajectory around earth."""
    fig = plt.figure(figsize=(18, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectory
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        color="black",
        label="Trajectory",
    )
    ax.scatter(
        trajectory[:1, 0],
        trajectory[:1, 1],
        trajectory[:1, 2],
        color="red",
        label="Initial Position",
    )

    # Plot planet
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]  # Spherical coordinates

    xs = EARTH_RADIUS * np.cos(u) * np.sin(v)
    ys = EARTH_RADIUS * np.sin(u) * np.sin(v)
    zs = EARTH_RADIUS * np.cos(v)
    ax.plot_surface(xs, ys, zs, cmap="Blues")

    # Plot arrows
    origin = np.zeros(3)
    length = EARTH_RADIUS * 2
    u, v, w = [[length, 0, 0], [0, length, 0], [0, 0, length]]
    ax.quiver(origin, origin, origin, u, v, w, color="black")

    # Configure plot settings
    max_ = np.max(trajectory)
    ax.set_xlim(-max_, max_)
    ax.set_ylim(-max_, max_)
    ax.set_zlim(-max_, max_)

    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")

    plt.title("2 Body System")
    plt.legend()

    # Show stuff
    plt.show()


def differential_equation(t, y, mu):
    """Differential equation for solving a restricted two-body problem.

    Note: time and velocity are not used as acceleration only depends on position and gravity.
    """
    # Unpack state:
    rx, ry, rz, vx, vy, vz = y

    # Radius vector and its norm
    radius = np.array([rx, ry, rz])
    norm = np.linalg.norm(radius)

    # Two body acceleration
    ax, ay, az = -radius * mu / norm**3

    return [vx, vy, vz, ax, ay, az]


def main():
    """Plot an orbit around earth."""
    # Initial conditions of orbit
    radius = EARTH_RADIUS + 500  # km
    velocity = np.sqrt(EARTH_MU / radius)  # Velocity of circular orbit

    # Initial position and velocity
    initial_radius = np.array([radius, 0, 0])
    initial_velocity = np.array([0, velocity, 0])

    # Time parameters
    length = 100 * 60  # seconds
    delta = 100  # seconds
    num_steps = int(np.ceil(length / delta))

    # Initialize states
    states = np.zeros((num_steps, 6))
    times = np.zeros((num_steps, 1))

    states[0, :3] = initial_radius
    states[0, 3:] = initial_velocity

    # Propagate orbit
    solver = ode(differential_equation)
    solver.set_integrator("lsoda")
    solver.set_initial_value(states[0], 0)
    solver.set_f_params(EARTH_MU)

    for step in range(1, num_steps):
        if not solver.successful():
            break

        solver.integrate(solver.t + delta)
        times[step] = solver.t
        states[step] = solver.y

    trajectory = states[:, :3]
    plot(trajectory)


if __name__ == "__main__":
    main()
