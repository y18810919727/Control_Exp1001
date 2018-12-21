import math

def f(t):
    a = 2.1482*((t-8.435) +
                math.sqrt(8078.4 + (t-8.435)*(t-8.435)))-120
    return 1.0/a


if __name__ == '__main__':
    print(f(25))
