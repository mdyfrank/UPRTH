file = 'user_friend.txt'
c = 0
with open(file, 'r') as f:
    line = f.readline().strip()
    while line != '':
        c += len(line.split())-1
        line = f.readline().strip()
print(c)