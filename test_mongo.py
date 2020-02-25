import pymongo
from pymongo import MongoClient

client = MongoClient('localhost', 9000)

db = client['my_database']


import datetime
post = {"author":   "Mike",
        "text":     "My first blog post!",
        "tags":     ["mongodb", "python", "pymongo"],
        "date":     datetime.datetime.utcnow()
}

posts = db.posts
post_id = posts.insert_one(post).inserted_id

print('...')