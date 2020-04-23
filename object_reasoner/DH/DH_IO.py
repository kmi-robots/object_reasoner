"""
Methods to interact with Data Hub API:
Requires key to access DH stored in .ini config file
"""
import configparser
import requests

config = configparser.ConfigParser()
config.read('DHauth.ini')
DHdict = config['DH_ACCESS']

def get_imgs():

    return


