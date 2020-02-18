import os

final = []
files = [f for f in os.listdir('2019 data/') if os.path.isfile(os.path.join('2019 data/', f))]
for f in files:
    final.append(f[:-4])

file = open('companies_updated.txt', 'w', newline="", encoding='utf-8')
for comp in final:
    if comp == final[-1]:
        file.write(comp)
    else:
        file.write(comp + '\n')
