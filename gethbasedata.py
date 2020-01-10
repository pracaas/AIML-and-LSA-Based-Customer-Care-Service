#!/usr/bin/env python

"""
Created on Thu Aug 24 15:23:15 2018

@author: prakash
"""


"""
Insert data into HBase with a Python script.

To create the table, first use the hbase shell. We are going to create a
namespace called "sample_data". The table for this script is called "rfic",
as we will be inserting Request for Information Cases from the City of
Indianapolis.

Our table will have only one column family named "data", and we are accepting
all table defaults.

Original data source
https://data.indy.gov/dataset/Request-for-Information-Cases/ts4b-8qa9

% hbase shell
hbase> create_namespace "sample_data"
hbase> create "sample_data:rfic", "data"
"""

#import csv
import happybase
import time
import quesanswer as qa

#batch_size = 1
host = "localhost"
namespace = "sample_data"
row_count = 0
start_time = time.time()
table_name = "faq"


# for inserting the default dataset to Hbase database

'''
import quesanswer as qa

ques , answer = qa.main()

print(len(ques))

for i in range (0,len(ques)):
    print( i,". " ,ques[i], "\n" ,answer[i],"\n\n")
'''

def connect_to_hbase():
    """ Connect to HBase server.

    This will use the host, namespace, table name, and batch size as defined in
    the global variables above.
    """
    conn = happybase.Connection(host = host,
        table_prefix_separator = ":")
    conn.open()
    batch = conn.table(table_name)
    #batch = table.batch(batch_size = batch_size)
    return conn , batch


def insert_row(table):
    """ Insert a row into HBase.

    Write the row to the batch. When the batch size is reached, rows will be
    sent to the database.

    Rows have the following schema:
        [ id, keyword, subcategory, type, township, city, zip, council_district,
          opened, closed, status, origin, location ]
    """
    #for i in range (0,len(ques)):
    #    table.put(str(i+1),{'question:ques':ques[i],'question:ansindex':str(i+1),'answer:ans':answer[i]})
    #    print("Row - ", i ," Updated")

def add_ques(table,ques,answer):
    """ Insert a row into HBase.

    Write the row to the batch. When the batch size is reached, rows will be
    sent to the database.

    Rows have the following schema:
        [ id, keyword, subcategory, type, township, city, zip, council_district,
          opened, closed, status, origin, location ]
    """
    def LastRowNo(table):
        gen = table.scan()
        
        return len(list(gen))
        
    def AnsRowFinder(answer):
        row = table.scan(filter ="ValueFilter( =, 'binaryprefix:%s' )" %answer)
        ky = None
        for x in row:
            ky = x[0].decode()
        return ky
        
        
    ans_row_val = AnsRowFinder(answer)
    #print(ans_row_val,"It is answer row value from Database")
    #nextrow = str(LastRowNo(table) + 1)
    nextrow = str(GetLastRowNo() + 1)
    
    print("add_ques")
    #for i in range (0,len(ques)):
    if ans_row_val:
        table.put(nextrow,{'question:ques':ques,'question:ansindex':ans_row_val})
        #table.put(ans_row_val,{'car:hyundai':ques,'car:carindex':'haha'})
        print("Only New Question : with old anser is updated :",ans_row_val)


    else:
        table.put(nextrow,{'question:ques':ques,'question:ansindex':nextrow,'answer:ans':answer})
        #table.put(nextrow,{'car:hyundai':ques,'car:carindex':nextrow})
        #print(ques,"=>",answer ,"New Question with new answer Updated")
        print(ques,"New question with new answer updated")
        #val = "New question\n"+  


def add_chat(user="empty",bot="empty"):
    nextru = GetLastRowNo('answer:user') +1

    try:                           
        #GetLastRowNo('answer:bot')
        conn,table = connect_to_hbase()
        table.put(str(nextru),{'answer:user':user,'answer:bot':bot})

    finally:
        print('closed')
        conn.close()
    print(nextru)

def RowFinder(answer):
    try:                   
        
        conn,table = connect_to_hbase()
        row = table.scan(filter ="ValueFilter( =, 'binaryprefix:%s' )" %answer)
        ro = []
        for x in row:
            ky = x[0].decode()
            ro.append(ky)
        return ro
    
    finally:
        print('closed')
        conn.close()
    
    

def retrieve_ques(table):
    def conv(Rvalue,ques,ansindex):
        key = []
        value =[]
        index = []
        for x in Rvalue:
            ky = x[0].decode()
            val = x[1][ques].decode()
            indx = x[1][ansindex].decode()
            key.append(ky)
            value.append(val)
            index.append(indx)
            
        #print(key)
        #print(value)
        #print(value)
        #print(index)
        return value ,index
    
    ques = b'question:ques'
    ansindex = b'question:ansindex'
    value = table.scan(columns=[ques,ansindex])
    return conv(value,ques,ansindex)

def retrieve_ans(table,index):
    
    rows = table.row(index)
    test='answer:ans'.encode()
    row = rows[test]
    return row.decode()
    
    ''' 
    def conv(Rvalue,quali):
        key = []
        value =[]
        for x in Rvalue:
            ky = x[0].decode()
            val = x[1][quali].decode()
            key.append(ky)
            value.append(val)
            
        #print(key)
        #print(value)
        return value
    
    quali = b'question:ques'
    value = table.scan(columns=[quali])
    return conv(value,quali)
    '''

def deleterow(row):
    try:                   
        
        conn,batch = connect_to_hbase()
        batch.delete(row,columns=[b'question:ques', b'answer:ans',b'question:ansindex'])
        print('Row',row, ' has been deleted sucessfully')
        
    finally:
        print('closed')
        #print("delete unsucessfull")
        conn.close()
        
def deleterowC(row):
    try:                   
        
        conn,batch = connect_to_hbase()
        batch.delete(row,columns=[b'answer:bot', b'answer:user'])
        print('Row',row, ' has been deleted sucessfully')
        
    finally:
        print('closed')
        #print("delete unsucessfull")
        conn.close()

def getdata(typ,index='1'):
    try:                   
        #row = ['3','Good aerodynamics']
        # After everything has been defined, run the script.
        conn,batch = connect_to_hbase()
        #print(type(row))
        #insert_row(batch)
        #print("Row value gettype :",index," ",typ)
        if 'ans' in typ:
            return retrieve_ans(batch,index)
        elif 'ques' in typ:
            return retrieve_ques(batch)
            
    finally:
        print('closed')
        conn.close()

def retrieveC(table):
    ques=[]
    ans=[]
    mrow = GetLastRowNo('answer:user')
    for i in range (1,mrow):    
        rows = table.row(str(i))
        us='answer:user'.encode()
        an='answer:bot'.encode()
        #print(rows)
        usr = rows[us]
        bt = rows[an]
        ques.append(usr.decode())
        ans.append(bt.decode())
    return ques, ans
        
def getdataC():
    try:                   
        conn,batch = connect_to_hbase()
        
        return retrieveC(batch)
            
    finally:
        print('closed')
        conn.close()


def GetLastRowNo(column='question:ques'):
    try:                   
        conn,table = connect_to_hbase()
        gen = table.scan(columns=[column])
        '''
        scan(row_start=None, row_stop=None, 
             row_prefix=None, columns=None, 
             filter=None, timestamp=None, 
             include_timestamp=False, 
             batch_size=1000, scan_batching=None, 
             limit=None, sorted_columns=False, reverse=False)
        '''
        return len(list(gen))
        
    finally:
        print('closed')
        conn.close()
        
def givedata(ques,answer):
    try:                   
        
        conn,batch = connect_to_hbase()
        add_ques(batch,ques,answer)
        
    finally:
        print('closed')
        conn.close()

def DeleteDbRange(a,b):
    for i in range (a,b):
        deleterow(str(i))

def DeleteDbRangeC(a,b):
    for i in range (a,b):
        deleterowC(str(i))


#DeleteDbRangeC(0,350)
#DeleteDbRange(0,350)
#ques, ans = qa.main()
#def aa():
#    for i in range (0,len(ques)):
#        q = ques[i]
#        a = ans[i]
#        givedata(q,a)            
#aa()

#print(GetLastRowNo('question:ques'))

#DeleteDbRange(282,290)
#duration = time.time() - start_time
#print ("Done. row count: %i, duration: %.3f s" % (row_count, duration))

#question, ansindex = getdata('ques')
#getdata('ans','281')
    