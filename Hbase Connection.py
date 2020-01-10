#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 18:10:09 2018

@author: prakash
"""
import happybase


connection = happybase.Connection('localhost',9090,autoconnect=False)

connection.open()

#print(connection.tables())

connection.create_table(
    'mytable',
    {'cf1': dict(max_versions=10),
     'cf2': dict(max_versions=1, block_cache_enabled=False),
     'cf3': dict(),  # use defaults
    }
)

table = connection.table('mytable')


print(connection.tables())

row = table.row(b'row-key')
print(row[b'cf1:col1']) 
