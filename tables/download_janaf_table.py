import requests
import sys

speciesname = sys.argv[1]
newname     = sys.argv[2]

url = 'https://janaf.nist.gov/tables/' + speciesname + '.txt'

webpage = requests.get(url)

open('./janaf_tables/' + newname + '.txt', 'wb').write(webpage.content)
