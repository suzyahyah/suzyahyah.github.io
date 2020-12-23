---
layout: post
title: "Recipe for building jq from source without admin(sudo) rights"
date: 2020-03-31
mathjax: true
status: []
categories: [Misc]
---


#### This took me some time to install. Just putting it out there in case it helps someone.

Get the latest jq from github

`git clone https://github.com/stedolan/jq.git`

Update submodules (onigurama)

`git submodule update --init`

Copy missing auxiliary files

`autoreconf -fi`

Install into {YOUR_HOME_DIR} with onigurama (regex library)

`./configure --with-oniguruma=builtin --disable-maintainer-mode --prefix={YOUR_HOME_DIR}.local`

Check that we have all the dependencies downloaded and install

`make -j8 && make check` 

`make install`

Finally, add this to your ~/.bashrc and to call jq from anywhere. Remember
to `>source ~/.bashrc` and you should be good to go.

`export PATH=$PATH:~/.local/bin/`
