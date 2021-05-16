import re
import random 
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
final_dict = open('./training/final_dict.txt', 'r')
o = open('./training/demotic-english.txt', 'w')

for line in d:
    o.write(line.strip() + '|' + e.readline().strip() + '\n')

for line in final_dict:
	o.write(line)



final_dict.close()
d.close()
e.close()
o.close()

lines = open('./training/demotic-english.txt').readlines()
random.shuffle(lines)
open('./training/demotic-english.txt', 'w').writelines(lines)