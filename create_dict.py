import pickle

with open('tickers_scrape_news.txt', 'r') as f:
    comps = f.read()
    data = comps.split('\n')

comp_dict = {}
index = 1
for c in data:
    if c == 'MAT.OQ' and index == 1:
        comp_dict['CA'] = c
        index += 1
        continue
    elif c == 'MAT.OQ' and index == 2:
        comp_dict['MAT'] = c
        continue
    elif c == 'VIACA.OQ':
        comp_dict['CBS'] = c
    elif c == 'TPR.N':
        comp_dict['COH'] = c
    else:
        comp_dict[c.split('.')[0]] = c

# Now both CA and MAT have the same value MAT.OQ, so we need to differentiate


pk_file = open('comp_dict.pkl', 'ab')
pickle.dump(comp_dict, pk_file)
pk_file.close()

inp = open('comp_dict.pkl', 'rb')
db = pickle.load(inp)
print(db)
