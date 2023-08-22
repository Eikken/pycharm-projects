__all__ = ("multi","add")
def add(a,b) :
    return a+b

def multi(a,b):
    return a*b

if __name__ == "__main__":
    print("Main 语句")
else:
    print(__name__,"普通语句")
#print("这是一条父类语句")