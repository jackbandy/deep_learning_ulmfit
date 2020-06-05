from fastai.text import URLs, untar_data

imdb_path = untar_data(URLs.IMDB)
print("IMDB: {}".format(imdb_path))

yelp_path = untar_data(URLs.YELP_REVIEWS)
print("YELP: {}".format(yelp_path))

ag_path = untar_data(URLs.AG_NEWS)
print("AG: {}".format(ag_path))

dbpedia_path = untar_data(URLs.DBPEDIA)
print("DBPEDIA: {}".format(dbpedia_path))
