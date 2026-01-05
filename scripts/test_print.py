def print(*args, **kwargs):
    __builtins__.print('prefix: ', end='')
    return __builtins__.print(*args, *kwargs)

if __name__ == "__main__":

    string = 'string to test'
    print(string)
    print('another string')
    print('and a final string')