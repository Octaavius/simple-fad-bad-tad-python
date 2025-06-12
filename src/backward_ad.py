class Node:
    def __init__(self, value, parents=(), op=''):
        self.value = value
        self.parents = parents  # List of (parent_node, local_grad)
        self.grad = 0.0
        self.op = op  # For visualization/debugging

    def backward(self, grad_output=1.0):
        self.grad += grad_output
        for parent, local_grad in self.parents:
            parent.backward(grad_output * local_grad)

    def __repr__(self):
        return f"Node(value={self.value}, grad={self.grad}, op={self.op})"


def add(a, b):
    out = Node(a.value + b.value, parents=[(a, 1.0), (b, 1.0)], op='+')
    return out

def mul(a, b):
    out = Node(a.value * b.value, parents=[(a, b.value), (b, a.value)], op='*')
    return out

def square(a):
    return mul(a, a)

def exp(a):
    import math
    out_val = math.exp(a.value)
    return Node(out_val, parents=[(a, out_val)], op='exp')


# Example usage:
if __name__ == "__main__":
    # Define inputs
    x = Node(3.0)  # Leaf node
    y = Node(2.0)  # Leaf node

    # Function: f = (x * y + x^2)
    f = add(mul(x, y), square(x))

    # Compute gradient of f w.r.t. x and y
    f.backward()

    print(f"f = {f.value}")
    print(f"df/dx = {x.grad}")
    print(f"df/dy = {y.grad}")
