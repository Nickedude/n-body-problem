"""Script to visualize the three body problem."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.constants import gravitational_constant
from scipy.integrate import ode


def plot(states: np.ndarray):
    """Animate the motion of the three bodies."""
    # TODO: Make 3D
    trajectory_1 = states[:, :3]
    trajectory_2 = states[:, 6:9]
    trajectory_3 = states[:, 12:15]

    fig, ax = plt.subplots()

    scatter_1 = ax.scatter(trajectory_1[0, 0], trajectory_1[0, 1])
    scatter_2 = ax.scatter(trajectory_2[0, 0], trajectory_2[0, 1])
    scatter_3 = ax.scatter(trajectory_3[0, 0], trajectory_3[0, 1])

    def update(frame):
        scatter_1.set_offsets(trajectory_1[frame : frame + 1, :2])
        scatter_2.set_offsets(trajectory_2[frame : frame + 1, :2])
        scatter_3.set_offsets(trajectory_3[frame : frame + 1, :2])

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=len(trajectory_1), interval=1
    )

    all_xs = np.stack((trajectory_1[:, 0], trajectory_2[:, 0], trajectory_3[:, 0]))
    all_ys = np.stack((trajectory_1[:, 1], trajectory_2[:, 1], trajectory_3[:, 1]))
    ax.set_xlim(all_xs.min(), all_xs.max())
    ax.set_ylim(all_ys.min(), all_ys.max())
    plt.show()


def compute_acceleration(mu: float, diff: np.ndarray) -> np.ndarray:
    """Compute the acceleration caused by the given body."""
    return -mu * diff / (np.linalg.norm(diff) ** 3)


def compute_total_acceleration(
    fst_mu: float, fst_diff: np.ndarray, snd_mu: float, snd_diff: np.ndarray
) -> np.ndarray:
    """Compute the acceleration exerted on a body by the two given bodies."""
    fst = compute_acceleration(fst_mu, fst_diff)
    snd = compute_acceleration(snd_mu, snd_diff)
    return fst + snd


def differential_equation(t, y, mu):
    """Differential equation for the three body problem."""
    r1, v1 = y[:3], y[3:6]
    r2, v2 = y[6:9], y[9:12]
    r3, v3 = y[12:15], y[15:]

    mu1, mu2, mu3 = mu

    r1r2 = r1 - r2
    r1r3 = r1 - r3
    a1 = compute_total_acceleration(mu2, r1r2, mu3, r1r3)

    r2r1 = r2 - r1
    r2r3 = r2 - r3
    a2 = compute_total_acceleration(mu3, r2r3, mu1, r2r1)

    r3r1 = r3 - r1
    r3r2 = r3 - r2
    a3 = compute_total_acceleration(mu1, r3r1, mu2, r3r2)

    return np.hstack((v1, a1, v2, a2, v3, a3))


def main():
    """Visualize a three body problem."""
    # Masses in kg
    m1 = m2 = m3 = 1 / gravitational_constant
    mu = np.array(
        [
            gravitational_constant * m1,
            gravitational_constant * m2,
            gravitational_constant * m3,
        ]
    )

    # Initial conditions of the three bodies
    r1 = np.array([-1 / 3, -4 / 9, 0])
    v1 = np.zeros(3)

    r2 = np.array([2 / 3, -4 / 9, 0])
    v2 = np.zeros(3)

    r3 = np.array([-1 / 3, 8 / 9, 0])
    v3 = np.zeros(3)

    # Time parameters
    length = 18  # seconds
    delta = 0.01  # seconds
    num_steps = int(np.ceil(length / delta))

    # Initialize states
    times = np.zeros(num_steps)
    states = np.zeros((num_steps, 18))
    states[0, :3] = r1
    states[0, 3:6] = v1
    states[0, 6:9] = r2
    states[0, 9:12] = v2
    states[0, 12:15] = r3
    states[0, 15:18] = v3

    # Propagate orbits
    solver = ode(differential_equation)
    solver.set_integrator("lsoda")
    solver.set_initial_value(states[0], 0)
    solver.set_f_params(mu)

    for step in range(1, num_steps):
        if not solver.successful():
            break

        solver.integrate(solver.t + delta)
        times[step] = solver.t
        states[step] = solver.y

    plot(states)


if __name__ == "__main__":
    main()
