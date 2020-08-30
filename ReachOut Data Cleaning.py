import pandas as pd
import re

def main():
    print("This module contains functions for cleaning ReachOut forum posts")

def clean_RO_posts(posts_ls):
    """Clean ReachOut forum posts"""
    clean_posts_ls = []

    URL_format = re.compile("http[s]{0,1}://\S+", re.UNICODE)
    username_format = re.compile("@[A-Za-z0-9-_]+", re.UNICODE)

    for post in posts_ls:
        no_whitespace_text = post.replace('\n', '').replace('\xa0',' ').replace('\'', "'").replace('//', '')
        no_user_text = re.sub(username_format, '', no_whitespace_text)
        no_URL_text = re.sub(URL_format, '', no_user_text)
        no_backslash_text = re.sub("\'", "'", no_URL_text)
        no_ws_text = " ".join(no_backslash_text.split())
        clean_posts_ls.append(no_ws_text)
    
    return clean_posts_ls

def remove_moderators(clean_posts_ls, title_ls, username_ls, rank_ls, kudos_ls, date_ls, time_ls):
    """Automatically generate submissions from a subreddit"""
    df = pd.concat([pd.Series(clean_posts_ls, name='Clean_Post'), pd.Series(title_ls, name='Title')
                    , pd.Series(username_ls, name='Username'), pd.Series(rank_ls, name = "Rank")
                    , pd.Series(kudos_ls, name = "Kudos"), pd.Series(date_ls, name = "Date")
                    , pd.Series(time_ls, name = "Time")], axis = 1)
    
    length_post = [len(post) for post in clean_df['Clean_Post']]
    df.insert(2, "length_post", length_post)
    
  
    # Create User DF
    user_df = df[(df['Rank'] != 'Mod Squad') & (df['Rank'] != 'Mod') & (df['Rank'] != 'Post Mod')
                    & (df['Rank'] != 'Staff') & (df['Rank'] != 'Community Manager')]

    # Create Moderator DF
    mod_df = df[(df['Rank'] == 'Mod Squad') | (df['Rank'] == 'Mod') | (df['Rank'] == 'Post Mod')
                    | (df['Rank'] == 'Staff')]
    
    return user_df, mod_df

if __name__ == '__main__':   
    main()

