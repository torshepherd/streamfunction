import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray

# Define stream functions
def vortex(
    at_point: NDArray[np.float64],
    to_goal: NDArray[np.float64],
    obstacle: NDArray[np.float64],
    radius: float,
) -> NDArray[np.float64]:
    """
    Calculates the vortex function at a point.

    Parameters
    ----------
    at_point : ndarray (n_points, 2)
        The point at which to calculate the vortex function.
    to_goal : ndarray (2,)
        The goal point.
    obstacle : ndarray (2,)
        The cylindrical obstacle.
    radius : float
        The radius of the obstacle.

    Returns
    -------
    ndarray
        The vortex function at the point.
    """
    assert at_point.ndim == 2
    assert at_point.shape[1] == 2
    assert to_goal.shape == (2,)
    assert obstacle.shape == (2,)
    assert radius > 0

    a = radius
    bx = obstacle[0]
    by = obstacle[1]
    x = at_point[:, 0]
    y = at_point[:, 1]

    vectors_to_obstacle = obstacle - at_point
    ro = np.linalg.norm(vectors_to_obstacle, axis=1)

    denominator = (
        (a**4)
        + (2 * a**2 * (bx * (x - bx) + by * (y - by)))
        + (bx**2 + by**2) * ro**2
    )

    numerator_u = (
        bx * (x - bx) ** 2
        + (a**2) * (x - bx)
        + (y - by) * (2 * by * x - bx * (y + by))
    )
    numerator_v = (
        by * (y - by) ** 2
        + (a**2) * (y - by)
        + (x - bx) * (2 * bx * y - by * (x + bx))
    )

    u = (a**2 / ro**2) * numerator_u / denominator
    v = (a**2 / ro**2) * numerator_v / denominator

    return np.column_stack((u, v))


def sink(
    at_point: NDArray[np.float64],
    to_goal: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculates the sink function at a point.

    Parameters
    ----------
    at_point : ndarray (n_points, 2)
        The point at which to calculate the sink function.
    to_goal : ndarray (2,)
        The goal point.

    Returns
    -------
    ndarray
        The sink function at the point.
    """
    assert at_point.ndim == 2
    assert at_point.shape[1] == 2
    assert to_goal.shape == (2,)

    vectors_to_goal = to_goal - at_point
    return vectors_to_goal / np.sum(vectors_to_goal**2, axis=1, keepdims=True)


def stream(
    at_point: NDArray[np.float64],
    to_goal: NDArray[np.float64],
    obstacles: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Calculates the stream function at a point.

    Parameters
    ----------
    at_point : ndarray (n_points, 2)
        The point at which to calculate the stream function.
    to_goal : ndarray (2,)
        The goal point.
    obstacles : ndarray (n_obstacles, 3) (x, y, radius)
        The obstacles.

    Returns
    -------
    ndarray
        The stream function at the point.
    """
    assert at_point.ndim == 2
    assert at_point.shape[1] == 2
    assert to_goal.shape == (2,)
    assert obstacles.ndim == 2
    assert obstacles.shape[1] == 3

    # Add components to field
    field = sink(at_point, to_goal)
    for obstacle in obstacles:
        field += vortex(at_point, to_goal, obstacle[:2], obstacle[2])

    return field


if __name__ == "__main__":
    # Define the goal point
    to_goal = np.array([0, 0])

    # Define the obstacles
    obstacles = np.array(
        [
            [-9, 0.5, 0.7],
            [-7, -2, 0.7],
            [-6, 1, 0.7],
            [-4, -1, 0.7],
            [-3, 1, 0.7],
        ]
    )

    # Define the points at which to calculate the stream function
    SPACING = 100
    test_x, test_y = np.meshgrid(
        np.linspace(-11, 1, 12 * SPACING), np.linspace(-3, 3, 5 * SPACING)
    )
    test_points = np.column_stack((test_x.flatten(), test_y.flatten()))

    # Calculate the stream function
    field = stream(test_points, to_goal, obstacles)
    field_x = field[:, 0].reshape(test_x.shape)
    field_y = field[:, 1].reshape(test_y.shape)

    # Plot the stream function
    plt.figure(figsize=(10, 10))
    plt.streamplot(test_x, test_y, field_x, field_y, density=3)
    # plt.quiver(test_x, test_y, field_x, field_y)
    plt.scatter(obstacles[:, 0], obstacles[:, 1], s=obstacles[:, 2] * 20000, c="k")
    plt.scatter(to_goal[0], to_goal[1], c="r")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()
