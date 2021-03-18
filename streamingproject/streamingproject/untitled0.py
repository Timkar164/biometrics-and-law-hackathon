# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:59:35 2020

@author: pitonhik
"""

import pymongo
client = pymongo.MongoClient("localhost", 27017)
db = client.person
db.person.insert_one({'time': 1})
