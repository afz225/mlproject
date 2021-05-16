import re
f = open('./training/test_demotic.txt', 'r')
o = open('./training/processed_demotic.txt', 'w')

for line in f:
    o.write(re.sub('\A(\(.+?\))', '', line).strip()+'\n')

f.close()
o.close()

f = open('./training/test_english.txt', 'r')
o = open('./training/processed_english.txt', 'w')

for line in f:
    o.write(re.sub('\A(\(.+?\))', '', line).strip()+'\n')

f.close()
o.close()

d = open('./training/processed_demotic.txt', 'r')
e = open('./training/processed_english.txt', 'r')
o = open('./training/demotic-english.txt', 'w')

for line in d:
    o.write(line.strip() + '|' + e.readline().strip() + '\n')

d.close()
e.close()
o.close()