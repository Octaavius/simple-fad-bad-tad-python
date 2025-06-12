class ForwardAD:
    def __init__(self, value, derivative):
        self.value = value
        self.derivative = derivative

    def __add__(self, other):
        if isinstance(other, ForwardAD):
            return ForwardAD(self.value + other.value, self.derivative + other.derivative)
        else:
            return ForwardAD(self.value + other, self.derivative)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, ForwardAD):
            return ForwardAD(self.value * other.value, self.value * other.derivative + self.derivative * other.value)
        else:
            return ForwardAD(self.value * other, self.derivative * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, ForwardAD):
            return ForwardAD(self.value - other.value, self.derivative - other.derivative)
        else:
            return ForwardAD(self.value - other, self.derivative)

    def __rsub__(self, other):
        return ForwardAD(other - self.value, -self.derivative)

    def __truediv__(self, other):
        if isinstance(other, ForwardAD):
            return ForwardAD(self.value / other.value,
                             (self.derivative * other.value - self.value * other.derivative) / (other.value ** 2))
        else:
            return ForwardAD(self.value / other, self.derivative / other)

    def __rtruediv__(self, other):
        return ForwardAD(other / self.value, -other * self.derivative / (self.value ** 2))

    def __pow__(self, power):
        return ForwardAD(self.value ** power, power * self.value ** (power - 1) * self.derivative)

    def __repr__(self):
        return f"ForwardAD(value={self.value}, derivative={self.derivative})"


# Example of usage:
if __name__ == "__main__":
    x = ForwardAD(3, 1)  # x = 3, dx/dx = 1

    # Function: f(x) = x^2 + 2*x + 1
    f = x**2 + 2*x + 1

    print(f"Value of f(x) at x=3: {f.value}")  # f(x) = 3^2 + 2*3 + 1 = 16
    print(f"Derivative of f(x) at x=3: {f.derivative}")  # df/dx = 2*3 + 2 = 8
    # ---------------------------------------------------------------------------------
    x = ForwardAD(3, 1)  # x = 3, dx/dx = 1

    # Function: f(x) = (x + 1)(x - 2)
    #                     (x + 3) 
    f = ((x + 1) * (x - 2)) / (x + 3) 

    print(f"Value of f(x) at x=3: {f.value}")
    print(f"Derivative of f(x) at x=3: {f.derivative}")
