import mysql.connector


def fetchEmpInfo(empid):
    con=mysql.connector.connect(host='localhost',database='success200',user='root',password='')
    cur=con.cursor()
    try:
        fetch_qry="SELECT * from emp_entry where empid=(%s)"
        cur.execute(fetch_qry,(empid,))
        rec=cur.fetchall()
        row=list(rec[0])
        print(type(row[4]))
        return(row)
    except:
        print("notfount")
        return ["notfound","notfound"]

fetchEmpInfo("18IT002")
