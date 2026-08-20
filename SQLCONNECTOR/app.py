import mysql.connector
import pandas as pd

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="covid_db"
)

query = "SELECT * FROM covid_data"

df = pd.read_sql(query, conn)

print(df)

conn.close()