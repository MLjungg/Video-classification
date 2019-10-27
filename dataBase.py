import psycopg2

def connect():
    try:
        connection = psycopg2.connect(user="qmmtexwvhzczca",
                                      password='0e0f86784c9a57d1daac6ec57627e36a3a8f681db648d8a1e2cc79b56c9f87db',
                                      host="ec2-46-137-113-157.eu-west-1.compute.amazonaws.com",
                                      port = "5432",
                                      dbname = "d72t2upsejmv6v")
        cursor = connection.cursor()

        return cursor

    except (Exception, psycopg2.Error) as error:
        print(error)

def getVotes(db):
    query = 'SELECT "name","Vote" from "Votes","Videos" WHERE "Votes"."VideoId" = "Videos"."id" ORDER BY "name"'
    db.execute(query)

    return db.fetchall()
