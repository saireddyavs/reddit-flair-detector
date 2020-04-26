# reddit-fliae-detector
reddit flair detector


## Introduction
Red-Flair is a flair detector for posts belonging to [r/india](https://www.reddit.com/r/india/) subreddit on Reddit. Its web app is deployed using Heroku which can be accessed [here](https://reddit-flair-detector-saireddy.herokuapp.com/). It scrapes post's using the URL and then uses an Machine Learning Pipeline  to predict the flair of that post.

## How to use it


* **for single link**

   * As this project deals with only [reddit india](https://www.reddit.com/r/india/) you can pick any post from  [reddit india](https://www.reddit.com/r/india/) and copy the link of post and then paste it [here](https://reddit-flair-detector-saireddy.herokuapp.com/).Then it process it and gives you the  predicted output.
   
* ####  for multiple links

  * place multiple links in the text file one after the other only from [reddit india](https://www.reddit.com/r/india/).
  
  * save it on to disk.
  
  * we are going to predict for mutilple links using python [requests](https://requests.readthedocs.io/en/master/) .
  
  * what we are doing here.....
    
    * Sending request to the [server](https://reddit-flair-detector-saireddy.herokuapp.com/) with the text file attached in it which contains the links of posts from [reddit india](https://www.reddit.com/r/india/)
    
    * We will get json file as response.
    
    ```python
    
    import requests
    
    import json
    
    url="https://reddit-flair-detector-saireddy.herokuapp.com/automated_testing"
    
    files={'upload_file':open('location of your file in disk','rb')}
    
    response=requests.post(url,files=files)
    
    json_data = json.loads(response.text)
    ```
    
* **for running locally**

  * `git clone https://github.com/saireddyavs/reddit-flair-detector/` to clone the repository locally
  * `pip install -r requirements.txt` to install all the dependencies
  *  `python app.py` for starting the server
 
      * for single links open `http://127.0.0.1:5000` on your'e browser and copy paste the link from [reddit india](https://www.reddit.com/r/india/).
      
      * for mutiple links follow the [above code](#for-multiple-links) Except change in url to `http://127.0.0.1:5000/auomated_testing`.
      
    
    
    
  




