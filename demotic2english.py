d = open('./training/demotic.txt', 'r')
e = open('./training/english2.txt', 'r')
dict = open('./training/dict.txt', 'w')

for line in d:
    dict.write(line.strip()+'|'+e.readline().strip()+'\n')

dict.close()
d.close()
e.close()