# 1. Using python requests, perform an HTTP GET on a list of 10 webpages.                                                                                                                                                                 │····························
# 2. Aggregate the status codes and print them.                                                                                                                                                                                           │····························
import requests                                                                                                                                                                                                                           │····························

list_of_webpages = ['https://bbc.co.uk', 'https://nytimes.com', 'https://reddit.com', 'https://facebook.com',
                    'https://amazon.com', 'https://stackoverflow.com', 'https://youtube.com', 'https://berkeley.edu',
                    'https://stanford.edu']
# Import all libraries                                                                                                                                                                                                                    │····························

for webpage in list_of_webpages:                                                                                                                                                                                                          │····························
    get_http = requests.get(str(webpage))                                                                                                                                                                                             │····························
