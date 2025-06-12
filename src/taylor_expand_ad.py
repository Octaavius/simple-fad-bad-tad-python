import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, lambdify, Add, Mul, Pow, Integer
import itertools

# === Symbolic Setup ===
x, y = symbols('x y')

# === Utility Functions ===

def evaluate_atom(expr, x_coeffs, y_coeffs):
    if expr == x:
        return x_coeffs
    elif expr == y:
        return y_coeffs
    else:
        val = expr.subs({x: x_coeffs[0], y: y_coeffs[0]})
        return [val]

def convolve(p, q):
    max_order = max(len(p), len(q))
    return [
        sum((p[j] if j < len(p) else 0) * (q[i - j] if i - j < len(q) else 0)
            for j in range(i + 1))
        for i in range(max_order)
    ]

def add_lists(p, q):
    return [p[i] + q[i] if i < len(p) and i < len(q)
            else (p[i] if i < len(p) else q[i])
            for i in range(max(len(p), len(q)))]

def polynomial_product(p, q):
    if len(p) > len(q):
        return p[-1] * q[0]
    if len(p) < len(q):
        return q[-1] * p[0]
    return sum(p[i] * q[-(i + 1)] for i in range(len(p)))

def sympy_to_python(obj):
    if isinstance(obj, Integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: sympy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sympy_to_python(v) for v in obj]
    return obj

# === Taylor Expansion Infrastructure ===

def split_expr(expr, x_coeffs, y_coeffs, coefficients, node_counter):
    node_id = next(node_counter)

    if expr.is_Atom:
        coeff = evaluate_atom(expr, x_coeffs, y_coeffs)
        coefficients[node_id] = coeff
        return {"node_id": node_id, "node": str(expr), "coefficient": coeff}

    if isinstance(expr, Mul):
        left = split_expr(expr.args[0], x_coeffs, y_coeffs, coefficients, node_counter)
        right = split_expr(Mul(*expr.args[1:]) if len(expr.args) > 2 else expr.args[1],
                           x_coeffs, y_coeffs, coefficients, node_counter)
        coeff = convolve(left["coefficient"], right["coefficient"])
        coefficients[node_id] = coeff
        return {"node_id": node_id, "action": "mul", "left": left, "right": right, "coefficient": coeff}

    if isinstance(expr, Add):
        left = split_expr(expr.args[0], x_coeffs, y_coeffs, coefficients, node_counter)
        right = split_expr(Add(*expr.args[1:]) if len(expr.args) > 2 else expr.args[1],
                           x_coeffs, y_coeffs, coefficients, node_counter)
        coeff = add_lists(left["coefficient"], right["coefficient"])
        coefficients[node_id] = coeff
        return {"node_id": node_id, "action": "add", "left": left, "right": right, "coefficient": coeff}

    if isinstance(expr, Pow):
        base, exp = expr.args
        if exp == 0:
            coefficients[node_id] = [1]
            return {"node_id": node_id, "node": "1", "coefficient": [1]}
        left = split_expr(base, x_coeffs, y_coeffs, coefficients, node_counter)
        right = split_expr(Pow(base, exp - 1), x_coeffs, y_coeffs, coefficients, node_counter)
        coeff = convolve(left["coefficient"], right["coefficient"])
        coefficients[node_id] = coeff
        return {"node_id": node_id, "action": "mul", "left": left, "right": right, "coefficient": coeff}

    val = expr.subs({x: x_coeffs[0], y: y_coeffs[0]})
    coefficients[node_id] = [val]
    return {"node_id": node_id, "node": str(expr), "coefficient": [val]}

def next_coefficient(node, coefficients):
    if "action" not in node:
        return node["coefficient"]

    left_val = next_coefficient(node["left"], coefficients)
    right_val = next_coefficient(node["right"], coefficients)

    if node["action"] == "add":
        val = left_val[-1] + right_val[-1]
    else:
        val = polynomial_product(left_val, right_val)

    coefficients[node["node_id"]].append(val)
    return coefficients[node["node_id"]]

def taylor_expand(expr1, expr2, x0, y0, order):
    x_coeffs, y_coeffs = [x0], [y0]
    coefficients = {}
    node_counter = itertools.count()

    tree1 = split_expr(expr1, x_coeffs, y_coeffs, coefficients, node_counter)
    tree2 = split_expr(expr2, x_coeffs, y_coeffs, coefficients, node_counter)

    x_coeffs.append(tree1["coefficient"][0])
    y_coeffs.append(tree2["coefficient"][0])

    for i in range(order - 2):
        next_coefficient(tree1, coefficients)
        next_coefficient(tree2, coefficients)
        x_coeffs.append(tree1["coefficient"][-1] / (i + 2))
        y_coeffs.append(tree2["coefficient"][-1] / (i + 2))

    return x_coeffs, y_coeffs

# === Plotting Functions ===

def plot_combined(expr1, expr2, x_coeffs, y_coeffs, point, F_func=None, contour_level=None, zoom_out=5):
    f = lambdify((x, y), expr1, modules='numpy')
    g = lambdify((x, y), expr2, modules='numpy')

    x_vals = np.linspace(-zoom_out, zoom_out, 200)
    y_vals = np.linspace(-zoom_out, zoom_out, 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    U, V = f(X, Y), g(X, Y)
    mag = np.sqrt(U**2 + V**2)
    U_norm, V_norm = U / (mag + 1e-5), V / (mag + 1e-5)

    t = np.linspace(-2, 2, 500)
    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U_norm, V_norm, color='lightgray', alpha=0.4)

    for n in range(3, len(x_coeffs) + 1, 3):
        xt = sum(c * t**i for i, c in enumerate(x_coeffs[:n]))
        yt = sum(c * t**i for i, c in enumerate(y_coeffs[:n]))
        plt.plot(xt, yt, linewidth=0.5 + 1.5 * (n / len(x_coeffs)), alpha=0.1 + 0.9 * (n / len(x_coeffs)), label=f'Taylor {n}')

    if F_func and contour_level is not None:
        Z = F_func(X, Y)
        cp = plt.contour(X, Y, Z, levels=[contour_level], colors='purple')
        plt.clabel(cp, fmt=f"F(x,y)={contour_level}", fontsize=8)
        contour_proxy = plt.Line2D([0], [0], color='purple', linestyle='--', label=f"F(x,y) = {contour_level}")
        plt.gca().add_artist(contour_proxy)  

    plt.plot(*point, 'ko', label=f'Start {point}')
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Combined Phase Plot")
    plt.grid(True); plt.axis('equal'); plt.legend()
    plt.xlim(-zoom_out, zoom_out)
    plt.ylim(-zoom_out, zoom_out)
    plt.show()

# === Example Usage ===

expr1 = x * (x + y)
expr2 = y * (x - y)
start_point = (1, 1)

x_coeffs, y_coeffs = taylor_expand(expr1, expr2, start_point[0], start_point[1], order=12)

def F(x, y):
    with np.errstate(divide='ignore', invalid='ignore'):
        return -x / y + np.log(y) + np.log(x)

plot_combined(expr1, expr2, x_coeffs, y_coeffs, point=start_point, F_func=F, contour_level=-1, zoom_out=5)
