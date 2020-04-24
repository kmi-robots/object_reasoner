"""
Methods to interact with Data Hub API:
Requires key to access DH stored in .ini config file
"""
import configparser
import requests
from requests.auth import HTTPBasicAuth
import json
import datetime
import os
import time

config = configparser.ConfigParser()
config.read('DHauth.ini')
DHdict = config['DH_ACCESS']
complete_url = os.path.join(str(DHdict['url']),'sciroc-episode12-image')

def get_imgs():
    print(type(DHdict['url']))
    t = datetime.datetime.fromtimestamp(time.time())
    timestamp = t.strftime("%Y-%m-%dT%H:%M:%SZ")

    json_body= {
        "@id": "string",
        "@type": "ImageReport",
        "team": DHdict['teamid'],
        "timestamp": timestamp,
        }

    return requests.request("GET", complete_url, data=json.dumps(json_body),
                            auth=HTTPBasicAuth(DHdict['teamkey'], ''))


def get_local():

    with open(str(DHdict['local'])) as file:
        j = json.load(file)
        print(j[-1])
    return j

if __name__ == "__main__":

    print(get_local())
