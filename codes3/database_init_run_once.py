# C:\ProgramData\MySQL\MySQL Server 5.7\my.ini
# D:\ProgramData\MySQL\MySQL Server 5.7\Data
import mysql.connector

db = mysql.connector.connect(
    host ='localhost',
    user ='root',
    passwd ='password123',
    database ='blimuw',
    )
cursor = db.cursor()

cursor.execute("DROP TABLE IF EXISTS designs;")
db.commit()

# Add Table
cursor.execute("CREATE TABLE designs " \
                + "(" \
                    + "id INTEGER AUTO_INCREMENT PRIMARY Key, " \
                    + "name VARCHAR(255), " \
                        + "PS_or_SC VARCHAR(55), " \
                        + "DPNV_or_SEPA VARCHAR(55), " \
                        + "p INT, " \
                        + "ps INT, " \
                        + "MecPow FLOAT, " \
                        + "Freq FLOAT, " \
                        + "Voltage FLOAT, " \
                        + "TanStress FLOAT, " \
                        + "Qs INT, " \
                        + "Qr INT, " \
                        + "Js FLOAT, " \
                        + "Jr FLOAT, " \
                        + "Coil VARCHAR(55), " \
                        + "kCu FLOAT, " \
                        + "Condct VARCHAR(55), " \
                        + "kAl FLOAT, " \
                        + "Temp FLOAT, " \
                        + "Steel VARCHAR(55), " \
                        + "kFe FLOAT, " \
                        + "Bds FLOAT, " \
                        + "Bdr FLOAT, " \
                        + "Bys FLOAT, " \
                        + "Byr FLOAT, " \
                        + "G_b FLOAT, " \
                        + "G_eta FLOAT, " \
                        + "G_PF FLOAT, " \
                        + "debug BOOL, " \
                        + "Sskew BOOL, " \
                        + "Rskew BOOL, " \
                        + "Pitch INT, " \
                    + "meshview BLOB, " \
                    + "comments VARCHAR(255), " \
                    + "createon TIMESTAMP" \
                + ")")

cursor.execute("SHOW TABLES")
for table in cursor:
    print(table)

# Get Column name
cursor.execute("SHOW columns FROM designs")
print ([column[0] for column in cursor.fetchall()])
