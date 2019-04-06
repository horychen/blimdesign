# C:\ProgramData\MySQL\MySQL Server 5.7\my.ini
# D:\ProgramData\MySQL\MySQL Server 5.7\Data
import mysql.connector

db = mysql.connector.connect(
    host ='localhost',
    user ='root',
    passwd ='password123',
    database ='blimuw',
    # auth_plugin='mysql_native_password', # install MySQL server older than 8.0 to avoid this problem
    )
cursor = db.cursor()

## Create database
# cursor.execute('CREATE DATABASE blimuw')
# cursor.execute('SHOW DATABASES')
# for _db in cursor:
#     print(_db)

## Add Table
# cursor.execute("CREATE TABLE designs (name VARCHAR(255), comments VARCHAR(255), meshview BLOB, createon DATETIME, design_id INTEGER AUTO_INCREMENT PRIMARY Key)")
# cursor.execute("SHOW TABLES")
# for table in cursor:
#     print(table)

## Add one record
# sql = 'INSERT INTO designs (name, comments) VALUES (%s, %s)'
# record = ('p1ps2Qs24Qr16M19', 'For NineSigma proposal.')
# cursor.execute(sql, record)
# db.commit()

## Add many records
# sql = 'INSERT INTO designs (name, comments) VALUES (%s, %s)'
# records = [
#     ('p1ps2Qs24Qr16M19',)
#     (,)
#     (,)
# ]
# cursor.executemany(sql, record)
# db.commit()

## Select, fetch and inspect
cursor.execute('SELECT * FROM designs')
result = cursor.fetchall()
for row in result:
    print (row)
cursor.execute('SELECT name FROM designs')
result = cursor.fetchall()
for row in result:
    print (row)

## Where clause
cursor.execute("SELECT * FROM designs WHERE comments LIKE '%NineSigma%' AND design_id>3")
result = cursor.fetchall()
for row in result:
    print (row)

cursor.execute("SHOW columns FROM designs")
print ([column[0] for column in cursor.fetchall()])


# ## add new column to table
# column_name = 'createon'
# cursor.execute("ALTER TABLE designs ADD %s DATETIME" % column_name)

## Update
# cursor.execute("UPDATE designs SET comments='NineSigma' WHERE design_id=4")
# db.commit()
# cursor.execute("UPDATE designs SET createon='2019/03/15' WHERE design_id=4")
# db.commit()

## LIMIT results
# cursor.execute("SELECT * FROM designs LIMIT 3 OFFSET 0")
# cursor.execute("SELECT * FROM designs ORDER BY name DESC")
# cursor.execute("SELECT * FROM designs ORDER BY name ASC")
# print ([el for el in cursor.fetchall()])

## DELETE record
# 

## Drop table 
# 'DROP TABLE IF EXISTS table_name'




