import re

f = open('./training/beinlich.txt', 'r')


list = []

# Loop filters the beinlich wordlist to seperate out the ancient egyptian and german translations only
for line in f:
    entry = []
    seperated = line.split('|')
    entry.append(seperated[0].replace('A', 'ꜣ').replace('j', 'i҆').replace('a', 'ꜥ').replace('H', 'ḥ').replace('x', 'ḫ').replace('X', 'ẖ').replace('S', 'š').replace('q', 'ḳ').replace('T', 'ṯ').replace('D','ḏ').replace('jj', 'y'))
    temp = seperated[3]
    if re.sub('\([^>]+\)', '', seperated[3]) != '':
        temp = re.sub('\([^>]+\)', '', seperated[3])

    if '[' in temp:
        temp = temp[temp.find('[')+1:temp.find(']')]+'\n'
    temp = temp.strip()
    if temp != '':
        temp = temp.replace('oe', 'ö').replace('ae', 'ä').replace('Ue', 'ü').replace('Oe', 'ö').replace('Ae', 'ä')

        if 'eue' not in temp:
            temp = temp.replace('ue', 'ü')
        entry.append(temp + '\n')
        list.append(entry)
f.close()

german = open('./training/german.txt', 'w')

# loop jots down german translations in order to be processed externally and translated to english
for entry in list:
    german.write(entry[1])

german.close()

# loop jots down demotic in order to be combined externally with english translations
demotic = open('./training/demotic.txt', 'w')
for entry in list:
    demotic.write(entry[0]+'\n')
demotic.close()

o = open('./training/dict.txt', 'r')

final_list = []

# Section processes lines so that words with multiple translation sin the same line are split into seperate lines so that no word has more than one translation per line
for line in o:
    entry = line.split('|')
    if ',' in entry[1]:
        temp = entry[1].split(',')
        for part in temp:
            final_list.append((entry[0] + "|" + part.strip()+'\n'))

    else:
        final_list.append(entry[0] + "|" + entry[1].strip() + '\n')

o.close()

final = open('./training/final_dict.txt', 'w')
for line in final_list:
    final.write(line)
final.close()
