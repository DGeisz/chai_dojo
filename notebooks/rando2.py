# %%
def g():
    a = 1

    def fib(x):
        if x <= 1:
            return x + a

        return fib(x - 1) + fib(x - 2)

    return fib
