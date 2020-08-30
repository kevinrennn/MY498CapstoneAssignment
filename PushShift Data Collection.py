import pandas as pd
import requests
import json
from psaw import PushshiftAPI


def main():
    print("This module contains functions for scraping PushShift for Reddit data")

def get_cheater_data(filename):
    '''Opens the PUBG cheater file and returns
    a dictionary with cheater IDs as keys and
    their cheating start date in datetime format as values.'''
    
    url = "https://api.pushshift.io/reddit/search/submission"
    params = {"subreddit": "depressed"}
    submissions = requests.get(url, params = params)

def crawl_page(subreddit: str, last_page = None):
    """Crawl a page of results from a given subreddit"""
    url = "https://api.pushshift.io/reddit/search/submission"
    params = {"subreddit": subreddit, "size": 500, "sort": "desc", "sort_type": "created_utc"}
    if last_page is not None:
        if len(last_page) > 0:
            # resume from where we left at the last page
            params["before"] = last_page[-1]["created_utc"]
        else:
            # the last page was empty, we are past the last page
            return []
    results = requests.get(url, params)
    if not results.ok:
        # something wrong happened
        raise Exception("Server returned status code {}".format(results.status_code))
    return results.json()["data"]

def crawl_subreddit(subreddit, max_submissions = 1000000):
    """Crawl submissions from a subreddit"""
    submissions = []
    last_page = None
    while last_page != []:# and len(submissions) < max_submissions:
        last_page = crawl_page(subreddit, last_page)
        submissions += last_page
        time.sleep(3)
        if len(submissions) % 10000 == 0:
            print(len(submissions))
    return submissions[:max_submissions]

def crawl_comments(subreddit, max_comments = 1500000):
    """Crawl comments from a subreddit"""
    api = PushshiftAPI()
    gen = api.search_comments(subreddit=subreddit)
    comments = []
    for c in gen:
        comments.append(str(c))
        if len(comments) % 10000 == 0:
            print(len(comments))
        if len(comments) >= max_comments:
            break
    if False:
        for c in gen:
            comments.append(str(c))
    df = pd.DataFrame([obj.d_ for obj in comments])
    return df
            
if __name__ == '__main__':   # Only executed if it is run as a script
    main()

