ps -ef | grep python | grep train | awk -F' ' '{print $2}' | xargs kill -9
