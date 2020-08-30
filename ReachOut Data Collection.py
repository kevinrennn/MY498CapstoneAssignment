import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup  

def main():
    print("This module contains functions for scraping ReachOut for forum data")

def get_page_name(url = 'https://forums.au.reachout.com/t5/Getting-Help/bd-p/Getting_Help'):
    """Automatically generate all Getting Help post links"""
    mainpage_ls = [url]
    for i in np.arange(2,65):
        mainpage = url + str(i)
        mainpage_ls.append(mainpage)
    return mainpage_ls

def create_page_links():
    """Automatically generate the consecutive page links for ReachOut forum posts"""
    all_links = []
    last_page = []
    for mainpage in mainpage_ls:
        url = mainpage
        page=requests.get(url)
        soup=BeautifulSoup(page.content,'lxml')
        # Links
        a_tags = soup.find_all('a', {'class':'page-link lia-link-navigation lia-custom-event'})
        original_links = [atag['href'] for atag in a_tags]
        relevant_links = original_links[0:-6]
        for link in relevant_links:
            all_links.append("https://forums.au.reachout.com"+link)

    for link in all_links:
        url = link
        page=requests.get(url)
        soup=BeautifulSoup(page.content,'lxml')
        # Last Page
        li_ls = soup.find_all('li')
        for i, l in enumerate(li_ls):
            if str(l).startswith('<li class="lia-paging-page-last lia-js-data-pageNum'):
                page_nr = l.text.replace('\n','')
                last_page.append(page_nr)
                break
            elif (i+1) == len(li_ls):
                last_page.append('1')
            else:
                pass
            
    return all_links, last_page

def crawl_forum():
    """Crawl through all posts and comments from the Getting Help forum"""
    title_ls=[]
    posts_ls=[]
    username_ls = []
    rank_ls = []
    date_ls = []
    time_ls = []
    kudos_ls = []

    for link, page_nr in zip(all_links[2:], last_page[2:]):
        page_iterator = [link]
        if int(page_nr)>1:
            for i in np.arange(2, int(page_nr)+1):
                page_link = link + "/page/" + str(i)
                page_iterator.append(page_link)

        # Crawl through all the pages
        for i, p in enumerate(page_iterator):
            url=p
            page=requests.get(url)
            soup=BeautifulSoup(page.content,'lxml')
            post_title = soup.find_all('title')
            posts = soup.find_all('div', {'class':'lia-message-body-content'})
            kudos = soup.find_all('span', {'class':'MessageKudosCount lia-component-kudos-widget-message-kudos-count'})
            names = soup.find_all('img', {'class':'lia-user-avatar-message'})
            ranks = soup.find_all('div', {'class':'lia-message-author-rank lia-component-author-rank lia-component-message-view-widget-author-rank'})
            datetimes = soup.find_all('span', {'class':'DateTime'})
            
            # Collect Post Title
            for t in post_title:
                title = t.text
            
            # Collect Post
            for t in posts:
                posts_ls.append(t.text) 
                title_ls.append(title)
                
            # Collect Username
            for name in names[0:-5]:
                try:
                    username_ls.append(name['title'])
                except:
                    username_ls.append("Anonymous")
                    
            # Collect User Rank
            for rank in ranks:
                rank_ls.append(rank.text.replace('\t','').replace('\n','')) 
                
            # Collect Data and Time
            for dt in datetimes:
                date_ls.append(dt.text.replace('\t','').replace('\n','')[:11])
                time_ls.append(dt.text.replace('\t','').replace('\n','')[11:])
            
            # Collect Kudos
            for kudo in kudos:
                kudos_ls.append(kudo.text.replace('\t','').replace('\n',''))
            
    return title_ls, posts_ls, username_ls, rank_ls, date_ls, time_ls, kudos_ls
            
if __name__ == '__main__':   
    main()

