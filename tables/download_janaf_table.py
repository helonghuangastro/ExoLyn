'''
A script to download the JANAF table from NIST
Usage: python3 download_janaf_table.py <species name in janaf table> <molecular name>
e.g. if you want to download the JANAF table for Al2O, you can use the following command:
python3 download_janaf_table.py Al-092 Al2O
'''
import requests
import sys

speciesname = sys.argv[1]
newname     = sys.argv[2]

url = 'https://janaf.nist.gov/tables/' + speciesname + '.txt'

webpage = requests.get(url)

open('./janaf_tables/' + newname + '.txt', 'wb').write(webpage.content)
