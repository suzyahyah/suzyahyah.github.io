---
layout: post
title: "Recipe for connecting to Google Drive from Remote Server"
date: 2022-05-10
mathjax: true
status: []
categories: [Code]
---

This recipe uses Python API to connect to Google Drive for personal use. When doing collaborative research we may want to upload results files for easy
viewing and sharing and google drive is often familiar to everyone. Step 1-4 is sufficient for connecting from a local machine, while Step 5 and 6 is for
connecting through a remote server. Google Cloud API has been around for a long time and this
recipe should be relatively stable - thanks to Anton for helping me figure out which
OAuthClient to use.

**Step 1. [Install the Google Client Library](https://developers.google.com/drive/api/quickstart/python)**

`pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib`

**Step 2. Clone/copy [quickstart.py](https://github.com/googleworkspace/python-samples/blob/master/drive/quickstart/quickstart.py) onto your remote server.**

`wget https://raw.githubusercontent.com/googleworkspace/python-samples/master/drive/quickstart/quickstart.py`

In order to allow authorization of the python program (app) to our google drive we need to set up the OAuth consent (Step 3), and then get the apps' credentials (Step 4).

**Step 3. Set up OAuth Consent**

In the Google Cloud Console go to APIs and Services > OAuth Consent Screen >
Select: External, Put your own email address as a Test User, Leave your app in testing phase. Don't need to set any specific scopes

**Step 4: [Create OAuth Client ID Credentials for a Desktop App](https://developers.google.com/workspace/guides/create-credentials#desktop-app) using the OAuth Consent**

OauthClient2.0 ID -> Desktop App

Once we have downloaded the `credentials_secret.json` onto our local machine, scp it into the remote
server. We then authenticate (basically sign ourselves in as the test user), which grants the
app access to our google drive. Google authorization server returns us an access token, which
contains a list of scopes (how much access the app has to our account).

The trick is modifying the `quickstart.py` script so that we can sign in on our local machine.

**Step 5: Modify the quickstart script**

Modify `quickstart.py` to [not launch a web browser on the remote server](https://stackoverflow.com/questions/54230127/open-the-authorization-url-without-opening-browser-python), but
instead return a URL that we can use to sign in on our local machine. 

`creds = flow.run_local_server(host='localhost', port=5000, open_browser=False)`

**Step 6: Tunnel a remote port from your server to your local Machine**

`ssh -L 5000:localhost:5000 <servername>`

Copy the URL that gets printed out on the terminal and paste it into your web browser on your
local computer. Sign in with your google account and we should get the `token.json` returned to
the remote directory where you launched the app. 

**Bonus: Sharing csv files for quickview and sharing**

`file_metadata = {'name': 'results.csv', 'parents': [folder_id]}` 
`mimetype = "text/csv"`


